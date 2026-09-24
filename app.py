import runpy

# Execute the package UI module as the Streamlit script on every rerun.
runpy.run_module("sports_edge.app", run_name="__main__")
