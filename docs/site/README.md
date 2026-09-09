# Project website

Static GitHub Pages website. Serve this folder with `python -m http.server 8765`
and open http://localhost:8765. No package install or external CDN is required.

Run `python docs/site/build_assets.py` from the repository root after changing
the source centerline exports or modeling figure. The staged assets are the only
research data included in the site; recording sessions are not part of the build.

The publication section awaits the author's official title, author list,
affiliations, abstract, publication URL and approved results. Do not substitute
the OpenCR paper's authors or results for this project's metadata.

To publish, commit this directory and `.github/workflows/project-pages.yml`
to the repository default branch, then select GitHub Actions in Settings >
Pages. Run the Project website workflow. It publishes only `docs/site`.
Expected URL: https://tanjingv.github.io/brnchus_robot_VLA/

The branch explorer uses the exported model centerlines. It is not a browser
MuJoCo runtime and does not report measured navigation performance.
