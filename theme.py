"""
Visual identity for the app.

Design concept: this is a *document intelligence* tool — the honest subject
is "turn private PDFs into a searchable, cited record." That points at a
dossier / evidence-room aesthetic rather than a generic chat-app look:

  - ink-dark chrome for the app "system" (sidebar, controls)
  - warm paper-toned surfaces where actual document content appears
    (assistant answers, which are effectively excerpts of a document)
  - citations rendered as numbered, stamped evidence cards — honest to
    what they are: a ranked sequence of retrieved evidence, not a random
    grab-bag, so the numbering carries real information

Tokens:
  ink        #10141C   page background
  panel      #1A2130   sidebar / control surfaces
  panel-2    #232B3D   hover / nested surfaces
  paper      #F3ECDA   assistant message + evidence-card background
  paper-line #E4D9BE   borders on paper surfaces
  gold       #C9973F   primary accent (buttons, focus, evidence numbers)
  gold-dim   #8A6B2E   hover state for gold
  ink-text   #E9E6DC   primary text on dark
  muted      #8891A3   secondary text on dark
  sage       #6E9277   positive (up-vote, connected)
  brick      #B25539   negative (down-vote, remove, error)

Type:
  display  'Source Serif 4'  — wordmark, assistant answer text (reads like
                                a document excerpt, not a chat bubble)
  body     'Inter'            — UI chrome, sidebar, buttons
  mono     'IBM Plex Mono'    — citation metadata: filenames, page/score

Note on CSS selectors: these target Streamlit's data-testid attributes,
which are reasonably stable across recent versions (1.3x+) but can shift
between major releases. If a specific element doesn't pick up styling
after a Streamlit upgrade, open browser dev tools, inspect the element,
and adjust the selector below — the rest of the theme is unaffected.
"""

THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --ink: #10141C;
    --panel: #1A2130;
    --panel-2: #232B3D;
    --paper: #F3ECDA;
    --paper-line: #E4D9BE;
    --gold: #C9973F;
    --gold-dim: #8A6B2E;
    --ink-text: #E9E6DC;
    --muted: #8891A3;
    --sage: #6E9277;
    --brick: #B25539;
}

html, body, [class*="stApp"] {
    background-color: var(--ink) !important;
    color: var(--ink-text);
    font-family: 'Inter', sans-serif;
}

/* ---------- Header / wordmark ---------- */
.doc-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.1rem;
}
.doc-header .seal {
    font-size: 1.6rem;
    color: var(--gold);
    line-height: 1;
}
.doc-header h1 {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--ink-text);
    letter-spacing: -0.01em;
    margin: 0;
}
.doc-subtitle {
    font-family: 'Inter', sans-serif;
    font-style: italic;
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0.15rem 0 1.4rem 0;
    border-bottom: 1px solid var(--panel-2);
    padding-bottom: 1.1rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--panel-2);
}
section[data-testid="stSidebar"] h1 {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    color: var(--ink-text);
}
section[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold) !important;
    border-bottom: 1px solid var(--panel-2);
    padding-bottom: 0.4rem;
    margin-top: 0.6rem !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {
    color: var(--muted);
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 6px;
    border: 1px solid var(--gold-dim);
    background-color: transparent;
    color: var(--gold);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background-color: var(--gold);
    color: var(--ink);
    border-color: var(--gold);
}
.stButton > button[kind="primary"] {
    background-color: var(--gold);
    color: var(--ink);
}

/* ---------- Chat area ---------- */
div[data-testid="stChatMessage"] {
    border-radius: 10px;
    border: 1px solid var(--panel-2);
    /* Apply the dark surface to every message first. This deliberately
       avoids depending on Streamlit's internal assistant-avatar markup,
       which varies between Streamlit versions. */
    background-color: var(--panel) !important;
    color: var(--ink-text) !important;
    animation: fadeIn 0.25s ease;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
    color: var(--ink-text) !important;
}
/* Assistant messages: paper surface — this IS the document content */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background-color: var(--panel);
    color: var(--ink-text);
    border-color: #313B50;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) li {
    font-family: 'Source Serif 4', serif;
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--ink-text);
}
/* User messages: ink surface with gold accent edge */
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background-color: var(--panel-2) !important;
    border-left: 3px solid var(--gold);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
    div[data-testid="stChatMessage"] { animation: none; }
}

div[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif;
}

/* ---------- Evidence / citation cards ---------- */
.evidence-card {
    background-color: var(--panel-2);
    border: 1px solid #39445B;
    border-top: 2px dashed var(--gold-dim);
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.55rem;
    color: var(--ink-text);
}
.evidence-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
    letter-spacing: 0.02em;
}
.evidence-num {
    color: var(--gold-dim);
    font-weight: 600;
    margin-right: 0.4rem;
}
.evidence-score {
    background-color: var(--gold);
    color: var(--ink);
    border-radius: 3px;
    padding: 0.05rem 0.4rem;
    font-weight: 600;
}
.evidence-snippet {
    font-family: 'Inter', sans-serif;
    font-size: 0.87rem;
    color: var(--ink-text);
    line-height: 1.45;
}

/* ---------- Document management cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--panel-2);
    border-color: var(--panel-2) !important;
    border-radius: 8px;
}

/* ---------- Alerts ---------- */
div[data-testid="stAlertContentSuccess"] { color: var(--sage); }
div[data-testid="stAlertContentError"] { color: var(--brick); }
</style>
"""


def render_evidence_card(rank: int, source: str, location: str, snippet: str, score) -> str:
    """Build the HTML for one numbered evidence/citation card."""
    score_html = f'<span class="evidence-score">{score}</span>' if score is not None else ""
    # Escape the bare minimum to avoid breaking the markup; content here is
    # our own retrieved chunk text, not untrusted external HTML.
    safe_snippet = (
        snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return f"""
    <div class="evidence-card">
        <div class="evidence-head">
            <span><span class="evidence-num">SRC {rank:02d}</span>{source} &middot; {location}</span>
            {score_html}
        </div>
        <div class="evidence-snippet">{safe_snippet}</div>
    </div>
    """
