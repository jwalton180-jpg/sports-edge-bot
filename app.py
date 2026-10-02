import runpy

# Production redeploy marker: 2026-10-02T20:00Z — selected-games compatibility hotfix
# Execute the package UI module as the Streamlit script on every rerun.
runpy.run_module("sports_edge.app", run_name="__main__")
