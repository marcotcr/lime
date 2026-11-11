"""The ``lime-stratified-demo`` directive: embeds an interactive demo.

Usage::

    .. lime-stratified-demo:: sampling
       :anchor: sampling-demo
       :fallback: demos/sampling/fallback.png
       :alt: Text for the fallback figure.

The demo's files live in ``_static/demos/<name>/``: ``demo.js``, ``demo.css``
and ``data.js`` (plus whatever data.js points to). The page gets a container
``<div class="limestrat-demo">`` that demo.js fills in. Without JavaScript, the
``<noscript>`` fallback figure is shown. The scripts and styles are added only
to the pages that use the directive.
"""

import hashlib
import html
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives


def _version(path):
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


class LimeStratifiedDemo(Directive):
    required_arguments = 1
    option_spec = {
        "anchor": directives.unchanged,
        "fallback": directives.unchanged,
        "alt": directives.unchanged,
    }

    def run(self):
        env = self.state.document.settings.env
        name = self.arguments[0]
        depth = env.docname.count("/")
        static = "../" * depth + "_static/"
        root = f"{static}demos/{name}/"
        version = _version(Path(env.srcdir) / "_static" / "demos" / name / "data.js")
        anchor = self.options.get("anchor", f"{name}-demo")
        alt = html.escape(self.options.get("alt", ""), quote=True)
        fallback = self.options.get("fallback")
        noscript = (f'<noscript><img class="limestrat-demo-fallback" '
                    f'src="{static}{fallback}" alt="{alt}"></noscript>'
                    if fallback else "")
        markup = (f'<div class="limestrat-demo" id="{anchor}" data-demo="{name}" '
                  f'data-root="{root}" data-version="{version}">{noscript}</div>')
        demos = getattr(env, "limestrat_demos", {})
        demos.setdefault(env.docname, set()).add(name)
        env.limestrat_demos = demos
        return [nodes.raw("", markup, format="html")]


def _add_assets(app, pagename, templatename, context, doctree):
    demos = getattr(app.env, "limestrat_demos", {}).get(pagename, ())
    for name in sorted(demos):
        # Sphinx appends a ?v=<checksum> to local static files itself.
        app.add_css_file(f"demos/{name}/demo.css")
        app.add_js_file(f"demos/{name}/demo.js", loading_method="defer")


def _purge(app, env, docname):
    getattr(env, "limestrat_demos", {}).pop(docname, None)


def _merge(app, env, docnames, other):
    mine = getattr(env, "limestrat_demos", {})
    mine.update(getattr(other, "limestrat_demos", {}))
    env.limestrat_demos = mine


def setup(app):
    app.add_directive("lime-stratified-demo", LimeStratifiedDemo)
    app.connect("html-page-context", _add_assets)
    app.connect("env-purge-doc", _purge)
    app.connect("env-merge-info", _merge)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
