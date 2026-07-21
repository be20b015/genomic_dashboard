"""
3D protein structure viewer (e.g. for AlphaFold / ESMFold PDB output),
embedded via the NGL.js viewer library over CDN.
"""

import json

import streamlit.components.v1 as components


def render_structure(pdb_text: str, height: int = 500):
    """
    pdb_text: raw contents of a .pdb file (as produced by AlphaFold/ESMFold).
    """
    safe_pdb = json.dumps(pdb_text)

    html = f"""
    <div id="viewport" style="width:100%; height:{height}px;"></div>
    <div id="ngl-status" style="font-family:sans-serif; font-size:12px; color:#888;"></div>
    <script src="https://cdn.jsdelivr.net/npm/ngl@2.2.1/dist/ngl.js"></script>
    <script>
      function initViewer() {{
        if (typeof NGL === "undefined") {{
          document.getElementById("ngl-status").textContent =
            "NGL library failed to load (check network/CDN access).";
          return;
        }}
        const pdbData = {safe_pdb};
        const stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});

        window.addEventListener("resize", () => stage.handleResize(), false);

        const blob = new Blob([pdbData], {{ type: "text/plain" }});
        stage.loadFile(blob, {{ ext: "pdb" }}).then(function (component) {{
          component.addRepresentation("cartoon", {{ colorScheme: "residueindex" }});
          component.autoView();
        }}).catch(function (err) {{
          document.getElementById("ngl-status").textContent = "Failed to load structure: " + err;
        }});
      }}
      // Guard against the script tag not being fully evaluated yet in some browsers
      if (typeof NGL !== "undefined") {{
        initViewer();
      }} else {{
        window.addEventListener("load", initViewer);
      }}
    </script>
    """
    components.html(html, height=height + 20, scrolling=False)


def render_structure_from_url(pdb_url: str, height: int = 500):
    """
    Convenience variant when you already have a hosted PDB URL
    (e.g. an AlphaFold DB entry) rather than raw text.
    """
    safe_url = json.dumps(pdb_url)

    html = f"""
    <div id="viewport" style="width:100%; height:{height}px;"></div>
    <div id="ngl-status" style="font-family:sans-serif; font-size:12px; color:#888;"></div>
    <script src="https://cdn.jsdelivr.net/npm/ngl@2.2.1/dist/ngl.js"></script>
    <script>
      function initViewer() {{
        if (typeof NGL === "undefined") {{
          document.getElementById("ngl-status").textContent =
            "NGL library failed to load (check network/CDN access).";
          return;
        }}
        const stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});
        window.addEventListener("resize", () => stage.handleResize(), false);
        stage.loadFile({safe_url}).then(function (component) {{
          component.addRepresentation("cartoon", {{ colorScheme: "residueindex" }});
          component.autoView();
        }}).catch(function (err) {{
          document.getElementById("ngl-status").textContent = "Failed to load structure: " + err;
        }});
      }}
      if (typeof NGL !== "undefined") {{
        initViewer();
      }} else {{
        window.addEventListener("load", initViewer);
      }}
    </script>
    """
    components.html(html, height=height + 20, scrolling=False)
