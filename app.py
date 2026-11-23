"""Streamlit counsellor UI for the Lead Qualification & Readiness Assistant.

Run with::

    streamlit run app.py

The "Lead workbench" page walks the assignment's development flow in order:
read lead history -> NLP extraction -> completeness scoring -> LLM summary
and follow-up questions -> save counsellor review.
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from leadqual import __project__, __version__
from leadqual.config import load_settings
from leadqual.db import Database, open_database
from leadqual.exceptions import LeadQualError
from leadqual.llm.client import QualificationLLM
from leadqual.llm.prompts import prompt_debug_bundle
from leadqual.models import FIELD_SPECS_BY_NAME
from leadqual.nlp.preprocess import nlp_backend_status
from leadqual.nlp.similarity import INTENT_DESCRIPTIONS, similarity_backend_status
from leadqual.pipeline import (
    pipeline_preview,
    qualify_lead,
    record_review,
    similar_past_leads,
)
from leadqual.reporting import (
    leads_needing_info,
    missing_field_report,
    overview_metrics,
    provider_usage,
    readiness_distribution,
    report_to_frame,
    review_summary,
)
from leadqual.scoring import explain_score
from leadqual.seed import seed_database

# --------------------------------------------------------------------------
# Visual tokens
#
# Readiness bands are a STATUS encoding, so they use the reserved status
# palette - never a categorical series colour. On this light surface the
# "warning" step sits below 3:1 contrast by design, so every band is shipped
# with a text label and a glyph: colour never carries the meaning alone.
# Single-series magnitude charts use one sequential hue and no legend.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SEQUENTIAL_HUE = "#2a78d6"

BAND_COLOURS = {
    "READY": "#0ca30c",       # status: good
    "NEEDS_INFO": "#fab219",  # status: warning
    "NOT_READY": "#d03b3b",   # status: critical
}
BAND_GLYPHS = {"READY": "●", "NEEDS_INFO": "◐", "NOT_READY": "○"}
BAND_ORDER = ["READY", "NEEDS_INFO", "NOT_READY"]

st.set_page_config(
    page_title="Lead Qualification Assistant",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------------------------------
# Shared resources
# --------------------------------------------------------------------------
#: Settings that may be supplied through Streamlit secrets when deployed.
_SECRET_KEYS = (
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LEADQUAL_PROVIDER", "LEADQUAL_MODEL",
    "LEADQUAL_GEMINI_MODEL", "LEADQUAL_MAX_TOKENS", "LEADQUAL_OFFLINE",
    "LEADQUAL_ALLOW_FALLBACK", "LEADQUAL_GEMINI_RETRIES", "LEADQUAL_DB_PATH",
)


def bridge_secrets() -> None:
    """Copy Streamlit secrets into the environment.

    Locally the key comes from a git-ignored ``.env``. On Streamlit Community
    Cloud there is no ``.env`` — the key is entered in the app's Secrets
    panel. Copying those into ``os.environ`` lets :func:`load_settings` stay
    a plain environment-variable reader, with no Streamlit dependency inside
    the ``leadqual`` package.

    A real environment variable always wins, so this never overrides a local
    ``.env``.
    """
    try:
        secrets = st.secrets
    except Exception:
        return  # no secrets.toml configured - normal when running locally
    for key in _SECRET_KEYS:
        try:
            if key in secrets and not os.getenv(key):
                os.environ[key] = str(secrets[key])
        except Exception:
            continue


@st.cache_resource(show_spinner="Preparing language resources...")
def ensure_nltk_data() -> dict[str, bool]:
    """Download the NLTK corpora if they are missing.

    A fresh Community Cloud container has none of them. The NLP layer falls
    back to regex and a built-in stopword list without them, but sentence
    segmentation and sentiment are better with them, so fetch them once per
    container. Failures are non-fatal.
    """
    results: dict[str, bool] = {}
    try:
        import nltk
    except ImportError:
        return results
    for package in ("punkt", "punkt_tab", "stopwords", "vader_lexicon"):
        try:
            results[package] = bool(nltk.download(package, quiet=True))
        except Exception:
            results[package] = False
    return results


@st.cache_resource(show_spinner=False)
def get_database(path: str) -> Database:
    """Open (once per session) the SQLite database.

    On a fresh deployment the database file does not exist — it is a
    generated artefact and is git-ignored — so an empty one is seeded with
    the synthetic sample data. Without this a deployed app would open on a
    blank dashboard.
    """
    database = open_database(path)
    try:
        if (database.query_pipeline_overview().get("total_leads") or 0) == 0:
            seed_database(database)
    except LeadQualError:
        pass  # a blank app is better than a crash; Settings can seed manually
    return database


def show_technical() -> bool:
    """Whether to show audit/debug panels (token counts, raw JSON, prompts).

    A counsellor does not need any of it; a marker and a developer do. The
    data is stored and queryable either way — this only controls how much of
    it is on screen. Analytics & audit always shows the full picture.
    """
    return bool(st.session_state.get("show_technical", False))


def band_label(band: str) -> str:
    """Status text with its glyph, so colour is never the only signal."""
    return f"{BAND_GLYPHS.get(band, '·')} {band.replace('_', ' ').title()}"


def band_badge(band: str, score: float) -> None:
    """Render a readiness badge with colour, glyph and text together."""
    colour = BAND_COLOURS.get(band, INK_MUTED)
    st.markdown(
        f"""
        <div style="
            display:inline-block;padding:0.45rem 0.9rem;border-radius:6px;
            background:{colour};color:#ffffff;font-weight:600;
            font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
            {BAND_GLYPHS.get(band, '·')} {band.replace('_', ' ')} &nbsp;·&nbsp; {score:.0f}/100
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def readiness_chart(frame: pd.DataFrame) -> alt.Chart:
    """Horizontal status bars: how many leads sit in each readiness band.

    Identity is carried by the axis label; colour is a redundant status cue,
    so there is no legend. Counts are direct-labelled at the bar end.
    """
    data = frame.copy()
    data["label"] = data["band"].map(band_label)

    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N",
            sort=[band_label(b) for b in BAND_ORDER],
            title=None,
            axis=alt.Axis(labelColor=INK_PRIMARY, labelFontSize=13, domain=False, ticks=False),
            scale=alt.Scale(paddingInner=0.35),
        ),
        x=alt.X(
            "lead_count:Q",
            title="Leads",
            axis=alt.Axis(
                grid=True, gridColor=GRIDLINE, tickCount=5,
                labelColor=INK_MUTED, titleColor=INK_MUTED, domain=False, ticks=False,
            ),
        ),
        tooltip=[
            alt.Tooltip("band:N", title="Band"),
            alt.Tooltip("lead_count:Q", title="Leads"),
            alt.Tooltip("share_pct:Q", title="Share %"),
            alt.Tooltip("avg_score:Q", title="Average score"),
        ],
    )

    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color(
            "band:N",
            scale=alt.Scale(
                domain=BAND_ORDER, range=[BAND_COLOURS[b] for b in BAND_ORDER]
            ),
            legend=None,
        )
    )
    labels = base.mark_text(
        align="left", dx=6, color=INK_PRIMARY, fontSize=12
    ).encode(text=alt.Text("lead_count:Q"))

    return (bars + labels).properties(height=150, background=SURFACE)


def missing_fields_chart(frame: pd.DataFrame) -> alt.Chart:
    """Single-series magnitude chart: how often each field is missing.

    One series, one hue, no legend - the title names the measure.
    """
    data = frame.head(10).copy()

    base = alt.Chart(data).encode(
        y=alt.Y(
            "field_label:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelColor=INK_PRIMARY, labelFontSize=12, domain=False, ticks=False),
            scale=alt.Scale(paddingInner=0.3),
        ),
        x=alt.X(
            "missing_pct:Q",
            title="% of leads where the field is missing",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(
                grid=True, gridColor=GRIDLINE, tickCount=5,
                labelColor=INK_MUTED, titleColor=INK_MUTED, domain=False, ticks=False,
            ),
        ),
        tooltip=[
            alt.Tooltip("field_label:N", title="Field"),
            alt.Tooltip("missing_pct:Q", title="Missing %"),
            alt.Tooltip("missing_count:Q", title="Leads missing it"),
            alt.Tooltip("total_count:Q", title="Leads assessed"),
            alt.Tooltip("is_required:N", title="Required"),
        ],
    )

    bars = base.mark_bar(cornerRadiusEnd=4, height=18, color=SEQUENTIAL_HUE)
    labels = base.mark_text(
        align="left", dx=6, color=INK_PRIMARY, fontSize=11
    ).encode(text=alt.Text("missing_pct:Q", format=".0f"))

    return (bars + labels).properties(height=max(200, 30 * len(data)), background=SURFACE)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_dashboard(db: Database) -> None:
    st.title("Dashboard")
    st.caption(
        "Information completeness across the enquiry pipeline. "
        "Every figure measures what is *on file*, never the enquirer."
    )

    metrics = overview_metrics(db)
    if not metrics.get("total_reports"):
        st.info(
            "No qualification reports yet. Load the sample data from "
            "**Settings & diagnostics**, then run a lead through the "
            "**Lead workbench**."
        )

    columns = st.columns(5)
    columns[0].metric("Leads", metrics.get("total_leads", 0))
    columns[1].metric("Messages", metrics.get("total_messages", 0))
    columns[2].metric("Reports", metrics.get("total_reports", 0))
    columns[3].metric("Avg readiness", f"{metrics.get('avg_score') or 0:.0f}/100")
    columns[4].metric("Awaiting review", metrics.get("awaiting_review", 0))

    left, right = st.columns(2)
    with left:
        st.subheader("Readiness of the pipeline")
        distribution = readiness_distribution(db)
        if distribution["lead_count"].sum():
            st.altair_chart(readiness_chart(distribution), width="stretch")
        else:
            st.caption("No scored leads yet.")
        with st.expander("Table view"):
            st.dataframe(distribution, width="stretch", hide_index=True)

    with right:
        st.subheader("What our enquiries fail to capture")
        missing = missing_field_report(db)
        if not missing.empty:
            st.altair_chart(missing_fields_chart(missing), width="stretch")
            worst = missing.iloc[0]
            st.caption(
                f"**{worst['field_label']}** is missing from "
                f"{worst['missing_pct']:.0f}% of assessed enquiries - the clearest "
                "candidate for a change to the enquiry form or first-reply template."
            )
        else:
            st.caption("No scored leads yet.")
        with st.expander("Table view"):
            st.dataframe(missing, width="stretch", hide_index=True)

    st.subheader("Counsellor work queue")
    st.caption("Leads whose latest assessment falls below the READY threshold.")
    queue = leads_needing_info(db)
    if queue.empty:
        st.success("Nothing in the queue - every assessed lead is READY.")
    else:
        st.dataframe(
            queue[
                ["lead_id", "full_name", "readiness_score", "readiness_band",
                 "open_followups", "missing_required"]
            ].rename(
                columns={
                    "lead_id": "Lead",
                    "full_name": "Name",
                    "readiness_score": "Score",
                    "readiness_band": "Band",
                    "open_followups": "Open questions",
                    "missing_required": "Missing required fields",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def page_workbench(db: Database, settings) -> None:
    st.title("Lead workbench")
    st.caption(
        "Read history → NLP extraction → completeness scoring → "
        "LLM summary & follow-up questions → counsellor review"
    )

    leads = db.list_leads(limit=500)
    if not leads:
        st.warning("No leads yet. Load the sample data from **Settings & diagnostics**.")
        return

    options = {
        f"#{row['lead_id']} · {row['full_name']} · {row['status']}"
        f" ({row['message_count']} msg)": row["lead_id"]
        for row in leads
    }
    chosen = st.selectbox("Choose a lead", list(options.keys()))
    lead_id = options[chosen]

    # ---- Step 1: read lead history ------------------------------------
    st.header("1 · Lead history")
    lead = db.get_lead(lead_id)
    messages = db.get_messages(lead_id)

    info_left, info_right = st.columns([2, 1])
    with info_left:
        st.markdown(
            f"**{lead['full_name']}** &lt;{lead['email']}&gt;  \n"
            f"Reference `{lead['external_ref'] or '-'}` · source *{lead['source']}* · "
            f"status *{lead['status']}*  \n"
            f"Programme on record: *{lead['programme_interest'] or 'not recorded'}*"
        )
    with info_right:
        latest = db.get_latest_report(lead_id)
        if latest:
            band_badge(latest["readiness_band"], float(latest["readiness_score"]))
            st.caption(f"Latest report #{latest['report_id']} · {latest['created_at']}")
        else:
            st.caption("Not yet qualified.")

    for message in messages:
        direction = "Enquirer" if message["direction"] == "inbound" else "Us"
        with st.chat_message("user" if message["direction"] == "inbound" else "assistant"):
            st.markdown(
                f"**{direction}** · {message['channel']} · {message['sent_at']}\n\n"
                f"{message['body']}"
            )

    with st.expander("Add another message to this lead"):
        with st.form(f"add_message_{lead_id}"):
            body = st.text_area("Message body", height=100)
            channel = st.selectbox(
                "Channel", ["email", "web_form", "phone", "whatsapp", "chat", "sms"]
            )
            direction = st.selectbox("Direction", ["inbound", "outbound"])
            if st.form_submit_button("Add message"):
                try:
                    from leadqual.validation import validate_message_payload

                    payload = validate_message_payload(
                        {"body": body, "channel": channel, "direction": direction},
                        lead_id=lead_id,
                    )
                    db.insert_message(**payload)
                    st.success("Message added.")
                    st.rerun()
                except LeadQualError as exc:
                    st.error(str(exc))

    # ---- Steps 2 & 3: NLP extraction and scoring ----------------------
    st.header("2 · NLP extraction and completeness scoring")
    st.caption("Deterministic and offline - no API call, nothing written to the database.")

    if st.button("Run NLP analysis", type="secondary", key=f"analyse_{lead_id}"):
        try:
            st.session_state[f"preview_{lead_id}"] = pipeline_preview(db, lead_id)
        except LeadQualError as exc:
            st.error(str(exc))

    preview = st.session_state.get(f"preview_{lead_id}")
    if preview:
        nlp, score = preview["nlp"], preview["score"]

        meta_cols = st.columns(4)
        meta_cols[0].metric("Readiness", f"{score.score:.0f}/100", band_label(score.band))
        meta_cols[1].metric("Intent", nlp.intent.replace("_", " "),
                            f"{nlp.intent_confidence:.2f} confidence")
        meta_cols[2].metric("Tone", nlp.sentiment, f"{nlp.sentiment_score:+.2f}")
        meta_cols[3].metric("Tokens analysed", nlp.token_count,
                            f"{nlp.message_count} inbound message(s)")

        st.caption(f"Intent meaning: {INTENT_DESCRIPTIONS.get(nlp.intent, '-')}")

        if nlp.protected_attribute_hits:
            st.warning(
                "This enquiry mentions personal details in these categories: **"
                + ", ".join(nlp.protected_attribute_hits)
                + "**. They are excluded from the score and redacted before any "
                "text reaches the language model. Handle them as a human judgement, "
                "not an automated one."
            )

        extraction = pd.DataFrame(
            [
                {
                    "Field": row["label"],
                    "Required": "yes" if row["required"] else "",
                    "Weight": row["weight"],
                    "Status": row["status"],
                    "Confidence": round(
                        nlp.fields[row["field"]].confidence if nlp.fields.get(row["field"]) else 0.0,
                        2,
                    ),
                    "Extracted value": (
                        nlp.fields[row["field"]].value or ""
                        if nlp.fields.get(row["field"]) else ""
                    ),
                }
                for row in explain_score(score)
            ]
        )
        st.dataframe(extraction, width="stretch", hide_index=True)

        if nlp.questions:
            st.markdown("**Questions the enquirer asked**")
            for question in nlp.questions:
                st.markdown(f"- {question}")

        if nlp.keywords:
            st.caption("Keywords: " + ", ".join(nlp.keywords))

        with st.expander("Similar past enquiries (TF-IDF cosine similarity)"):
            matches = similar_past_leads(db, lead_id, top_n=3)
            if matches:
                st.dataframe(pd.DataFrame(matches), width="stretch", hide_index=True)
            else:
                st.caption("No comparable enquiries on file yet.")

        if show_technical():
            with st.expander("The exact prompt that would be sent to the model"):
                bundle = prompt_debug_bundle(
                    preview["lead"], preview["messages"], nlp, score
                )
                st.caption(
                    f"System prompt: {bundle['system_chars']} characters (cached) · "
                    f"User turn: {bundle['user_chars']} characters"
                )
                st.text_area("System prompt", bundle["system"], height=240)
                st.text_area("User turn", bundle["user"], height=240)
                st.text_area("Response JSON schema", bundle["schema"], height=180)

    # ---- Step 4: LLM summary ------------------------------------------
    st.header("3 · Qualification summary and follow-up questions")
    provider_status = QualificationLLM(settings).status()
    if provider_status["live_calls_enabled"]:
        st.caption(
            f"Will make a live API call: **{provider_status['provider']}** / "
            f"`{provider_status['model']}`"
        )
    else:
        st.caption(f"Offline generator will be used — {provider_status['reason']}")

    if st.button("Generate summary and save report", type="primary", key=f"qualify_{lead_id}"):
        with st.spinner("Analysing the enquiry and generating the summary..."):
            try:
                st.session_state[f"report_{lead_id}"] = qualify_lead(
                    db, lead_id, settings=settings
                )
                st.session_state.pop(f"preview_{lead_id}", None)
            except LeadQualError as exc:
                st.error(str(exc))

    report = st.session_state.get(f"report_{lead_id}")
    if report:
        band_badge(report.readiness_band, report.readiness_score)
        meta = report.metadata
        if meta:
            if meta.is_fallback:
                st.warning(
                    "Generated by the **offline rule-based generator**, not a language "
                    f"model. Reason: {meta.note}"
                )
            elif show_technical():
                st.success(
                    f"Generated by **{meta.provider} / {meta.model}** · "
                    f"{meta.input_tokens} in / {meta.output_tokens} out tokens · "
                    f"{meta.latency_ms} ms · response `{meta.response_id}`"
                )
            else:
                # Provenance is always shown - a counsellor should know whether
                # a human-sounding summary came from a model - but the token
                # and request detail stays behind the technical toggle.
                st.success(f"Generated by **{meta.provider} / {meta.model}**")

        st.markdown("#### Summary for the counsellor")
        st.write(report.summary)

        if report.recommended_next_step:
            st.markdown("#### Recommended next step")
            st.info(report.recommended_next_step)

        st.markdown("#### Suggested follow-up questions")
        for followup in report.followups:
            label = (
                FIELD_SPECS_BY_NAME[followup.field_name].label
                if followup.field_name in FIELD_SPECS_BY_NAME
                else "general"
            )
            st.markdown(f"**{followup.priority}. {followup.question}**")
            st.caption(f"Fills: {label} — {followup.rationale}")

        if show_technical():
            with st.expander("Raw validated model payload (audit)"):
                st.json(report.llm_payload)

        # ---- Step 5: counsellor review --------------------------------
        st.header("4 · Counsellor review")
        with st.form(f"review_{report.report_id}"):
            reviewer = st.text_input("Your name", value="A. Counsellor")
            decision = st.selectbox(
                "Decision", ["approved", "edited", "rejected", "escalated"]
            )
            lead_status = st.selectbox(
                "Move the lead to",
                ["(leave unchanged)", "new", "in_progress", "qualified",
                 "needs_info", "closed"],
            )
            notes = st.text_area("Notes", height=90)
            if st.form_submit_button("Save review", type="primary"):
                try:
                    review_id = record_review(
                        db,
                        report.report_id,
                        reviewer=reviewer,
                        decision=decision,
                        notes=notes or None,
                        lead_status=None if lead_status.startswith("(") else lead_status,
                    )
                    st.success(
                        f"Review #{review_id} saved. Report #{report.report_id} "
                        "is now marked reviewed."
                    )
                except LeadQualError as exc:
                    st.error(str(exc))


def page_reports(db: Database) -> None:
    st.title("Qualification reports")

    rows = db.list_reports(limit=200)
    if not rows:
        st.info("No reports yet.")
        return

    frame = pd.DataFrame(rows)
    frame["is_fallback"] = frame["is_fallback"].astype(bool)
    st.dataframe(
        frame[
            ["report_id", "lead_id", "full_name", "readiness_score",
             "readiness_band", "status", "provider", "model", "is_fallback",
             "followup_count", "created_at"]
        ].rename(columns={"is_fallback": "offline_fallback"}),
        width="stretch",
        hide_index=True,
    )

    report_id = st.selectbox(
        "Open a report", [row["report_id"] for row in rows],
        format_func=lambda rid: f"Report #{rid}",
    )
    report = db.get_report(int(report_id))

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader(f"Report #{report['report_id']} — {report['full_name']}")
        st.caption(
            f"Created {report['created_at']} · status *{report['status']}* · "
            f"provider `{report['provider']}` / `{report['model']}`"
            + ("  ·  offline fallback" if report["is_fallback"] else "")
        )
    with header_right:
        band_badge(report["readiness_band"], float(report["readiness_score"]))

    st.markdown("#### Summary")
    st.write(report["summary"])
    if report.get("recommended_next_step"):
        st.info(report["recommended_next_step"])

    st.markdown("#### Qualification fields")
    st.dataframe(report_to_frame(report), width="stretch", hide_index=True)

    st.markdown("#### Follow-up questions")
    for followup in report["followups"]:
        cols = st.columns([6, 2, 2])
        cols[0].markdown(f"**{followup['priority']}.** {followup['question']}")
        cols[1].caption(f"status: {followup['status']}")
        if cols[2].button("Mark asked", key=f"ask_{followup['followup_id']}"):
            db.update_followup_status(followup["followup_id"], "asked")
            st.rerun()

    st.markdown("#### Counsellor reviews")
    if report["reviews"]:
        st.dataframe(
            pd.DataFrame(report["reviews"]), width="stretch", hide_index=True
        )
    else:
        st.caption("Not yet reviewed.")
        with st.form(f"review_existing_{report['report_id']}"):
            reviewer = st.text_input("Your name", value="A. Counsellor")
            decision = st.selectbox(
                "Decision", ["approved", "edited", "rejected", "escalated"]
            )
            notes = st.text_area("Notes", height=80)
            if st.form_submit_button("Save review"):
                try:
                    record_review(
                        db, int(report["report_id"]),
                        reviewer=reviewer, decision=decision, notes=notes or None,
                    )
                    st.success("Review saved.")
                    st.rerun()
                except LeadQualError as exc:
                    st.error(str(exc))

    if show_technical():
        with st.expander("Stored analysis and raw model payload (audit)"):
            st.json(
                {
                    "score": report.get("score"),
                    "llm_payload": report.get("llm_payload"),
                    "nlp": report.get("nlp"),
                }
            )


def page_analytics(db: Database) -> None:
    st.title("Analytics and audit")
    st.caption("SQL aggregates over the SQLite database, shaped with pandas.")

    st.subheader("Provider and model usage")
    st.caption(
        "Every report records which provider produced it, which model, the token "
        "usage and the latency — so a demonstration can always be traced back to "
        "a real API call or an offline fallback."
    )
    usage = provider_usage(db)
    if usage.empty:
        st.caption("No runs recorded yet.")
    else:
        st.dataframe(usage, width="stretch", hide_index=True)

    st.subheader("Counsellor decisions")
    reviews = review_summary(db)
    if reviews.empty:
        st.caption("No reviews recorded yet.")
    else:
        st.dataframe(reviews, width="stretch", hide_index=True)

    st.subheader("Audit log")
    entity = st.selectbox(
        "Entity type",
        ["(all)", "lead", "qualification_report", "counsellor_review", "database"],
    )
    audit = db.get_audit_log(
        entity_type=None if entity.startswith("(") else entity, limit=200
    )
    if audit:
        st.dataframe(pd.DataFrame(audit), width="stretch", hide_index=True)
    else:
        st.caption("Nothing logged yet.")


def page_settings(db: Database, settings) -> None:
    st.title("Settings & diagnostics")

    st.subheader("Sample data")
    st.caption(
        "The bundled dataset is entirely synthetic: invented names and "
        "`example.com` addresses only."
    )
    left, right = st.columns(2)
    if left.button("Load sample leads"):
        try:
            result = seed_database(db)
            st.success(
                f"Inserted {result['leads_inserted']} lead(s) and "
                f"{result['messages_inserted']} message(s); "
                f"skipped {result['leads_skipped']} already present."
            )
            for error in result["errors"]:
                st.warning(error)
        except LeadQualError as exc:
            st.error(str(exc))
    if right.button("Reset database and reload", type="secondary"):
        try:
            db.reset()
            result = seed_database(db)
            st.success(
                f"Database reset. Inserted {result['leads_inserted']} lead(s) and "
                f"{result['messages_inserted']} message(s)."
            )
            st.cache_resource.clear()
        except LeadQualError as exc:
            st.error(str(exc))

    st.subheader("Configuration")
    st.json(settings.describe())

    st.subheader("LLM provider")
    status = QualificationLLM(settings).status()
    if status["live_calls_enabled"]:
        st.success(status["reason"])
    else:
        st.warning(status["reason"])
    st.json(status)

    st.markdown("**All supported providers**")
    st.dataframe(
        pd.DataFrame(QualificationLLM.all_provider_status(settings)),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "To enable live calls: copy `.env.example` to `.env` and set "
        "`ANTHROPIC_API_KEY` (paid) or `GEMINI_API_KEY` "
        "(free — get one at aistudio.google.com/apikey), then restart the app. "
        "With no `LEADQUAL_PROVIDER` set, whichever key is present is used."
    )

    st.subheader("NLP backends")
    st.json({**nlp_backend_status(), **similarity_backend_status()})

    st.subheader("Database")
    st.json(db.query_pipeline_overview())
    st.caption(f"File: `{db.path}`")

    st.subheader("Fairness controls")
    st.markdown(
        """
The assistant assesses **information completeness only**. Three controls
enforce that in code rather than leaving it to the prompt:

1. **Protected attributes are detected** in the enquirer's own words
   (age, gender, marital or family status, religion, ethnicity or
   nationality, disability or health).
2. **They are redacted** from every piece of text before it reaches the
   language model, and from the keyword list shown above.
3. **They never affect the score.** The readiness score is a weighted
   completeness measure over the qualification fields; no protected
   attribute appears in that calculation.

The system prompt additionally forbids recommending that any enquirer be
deprioritised, declined or filtered out, and forbids treating a smaller
budget or a non-traditional background as a negative signal.
        """
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    # Must run before load_settings(): on Streamlit Community Cloud the API
    # key arrives as a secret, not as a .env file.
    bridge_secrets()
    ensure_nltk_data()

    settings = load_settings()

    st.sidebar.title("🎓 Lead Qualification")
    st.sidebar.caption(f"{__project__}  \nv{__version__}")

    db_path = st.sidebar.text_input("Database file", value=str(settings.db_path))
    try:
        db = get_database(db_path)
    except LeadQualError as exc:
        st.error(f"Could not open the database: {exc}")
        st.stop()
        return

    page = st.sidebar.radio(
        "Page",
        ["Dashboard", "Lead workbench", "Reports", "Analytics & audit",
         "Settings & diagnostics"],
    )

    st.sidebar.divider()
    # Declared before anything that reads it, so the toggle takes effect on
    # the same render rather than one behind.
    st.sidebar.checkbox(
        "Show technical details",
        key="show_technical",
        help=(
            "Provider and model names, token counts, latency, response ids, the "
            "raw model payload and the prompt inspector. Off by default: a "
            "counsellor does not need them, a marker or developer does. "
            "Everything is stored either way and always visible on "
            "Analytics & audit."
        ),
    )

    provider = QualificationLLM(settings).status()
    # The badge reports whether a live call is possible. Which provider and
    # model follows the technical toggle, like every other audit detail -
    # the per-report provenance line is what always names the model.
    if provider["live_calls_enabled"]:
        if show_technical():
            st.sidebar.success(
                f"Live API ready\n\n**{provider['provider']}**\n\n`{provider['model']}`"
            )
        else:
            st.sidebar.success("Live API connected")
    else:
        st.sidebar.warning("Offline generator\n\nNo API key configured")
    st.sidebar.caption(
        "Synthetic sample data only. The assistant assesses information "
        "completeness, never the enquirer."
    )

    if page == "Dashboard":
        page_dashboard(db)
    elif page == "Lead workbench":
        page_workbench(db, settings)
    elif page == "Reports":
        page_reports(db)
    elif page == "Analytics & audit":
        page_analytics(db)
    else:
        page_settings(db, settings)


if __name__ == "__main__":
    main()
