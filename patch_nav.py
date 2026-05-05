import re, os, glob

base = "/home/techpark-5/Downloads/Multi model AI agent-20260424T030916Z-3-001/Multi model AI agent/Medical AI/vaidyaai/stitch_vaidyaai_clinical_analysis_suite/stitch_vaidyaai_clinical_analysis_suite"

pages = glob.glob(os.path.join(base, "*/code.html"))

replacements = [
    (r'href="#"([^>]*)>Dashboard', r'href="/ui/vaidyaai_home/code.html"\1>Dashboard'),
    (r'href="#"([^>]*)>Jobs',      r'href="/ui/vaidyaai_analysis_status/code.html"\1>Jobs'),
    (r'href="#"([^>]*)>Insights',  r'href="/ui/vaidyaai_risk_insights/code.html"\1>Insights'),
    (r'href="#"([^>]*)>Reports',   r'href="/ui/vaidyaai_final_clinical_report_1/code.html"\1>Reports'),
    (r'href="#"([^>]*)>Analysis',  r'href="/ui/vaidyaai_analysis_status/code.html"\1>Analysis'),
]

SKIP = {"vaidyaai_upload_dashboard", "vaidyaai_analysis_status"}

for p in sorted(pages):
    folder = os.path.basename(os.path.dirname(p))
    if folder in SKIP:
        print(f"SKIP: {folder}")
        continue
    with open(p, "r", encoding="utf-8") as f:
        html = f.read()
    original = html
    for pattern, repl in replacements:
        html = re.sub(pattern, repl, html)
    # Wire "Start Analysis" CTA button -> upload page
    html = html.replace(
        '>Start Analysis\n                    </button>',
        " onclick=\"window.location.href='/ui/vaidyaai_upload_dashboard/code.html'\">\n                        Start Analysis\n                    </button>"
    )
    if html != original:
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"PATCHED: {folder}")
    else:
        print(f"NO CHANGE: {folder}")
