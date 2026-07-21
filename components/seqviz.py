"""
Interactive, zoomable DNA/RNA sequence viewer.

There isn't a maintained PyPI package that wraps the seqviz JS library
for Streamlit, so this embeds seqviz directly via CDN inside an
st.components.v1.html iframe. This is more reliable than depending on
an unverified package name.
"""

import json

import streamlit.components.v1 as components


def render_seqviz(sequence: str, name: str = "sequence", viewer_type: str = "linear", height: int = 400):
    """
    viewer_type: 'linear' or 'circular'
    """
    safe_seq = json.dumps(sequence)
    safe_name = json.dumps(name)
    safe_type = json.dumps(viewer_type)

    html = f"""
    <div id="seqviz-root" style="width:100%; height:{height}px;"></div>
    <script src="https://unpkg.com/seqviz/dist/seqviz.min.js"></script>
    <script>
      const seq = {safe_seq};
      const name = {safe_name};
      const viewerType = {safe_type};

      seqviz
        .Viewer("seqviz-root", {{
          name: name,
          seq: seq,
          viewer: viewerType,
          showComplement: true,
          showIndex: true,
          zoom: {{ linear: 50 }},
          style: {{ height: "{height}px", width: "100%" }},
        }})
        .render();
    </script>
    """
    components.html(html, height=height + 20, scrolling=True)
