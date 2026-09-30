import runpy

# Production redeploy marker: 2026-09-30T16:22Z — force Streamlit Cloud onto current main
# Execute the package UI module as the Streamlit script on every rerun.
runpy.run_module("sports_edge.app", run_name="__main__")
