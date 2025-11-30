# ComfyUI Workflow Placeholder

Import your tuned DiffuEraser + SAM2 ComfyUI workflow and save it to
`workflows/sora-removal-production.json` before running the pipeline CLI.

Recommended steps:

1. Open ComfyUI on the Windows rig and load the curated workflow in the GUI.
2. Confirm the graph includes the following blocks:
   - `DownloadAndLoadSAM2Model` (segmentor=`video`) feeding `Sam2VideoSegmentation*`
   - `Sam2VideoSegmentationAddPoints` → `Sam2VideoSegmentation` → `MaskToImage`
   - `DiffuEraserSampler` receiving the mask image
   - A `SaveImage` node consuming the inpainted frames
   The CLI will override the `coordinates_positive` field on
   `Sam2VideoSegmentationAddPoints` using the `--sam-point` argument you pass.
3. Verify the workflow ends with a `SaveImage` (frames) node so that the
   automation can download per-frame outputs.
4. Export the workflow JSON and commit it (or copy it) as
   `workflows/sora-removal-production.json` within this repository.
5. Keep any large model files out of git; only include the workflow graph.

Before running the CLI or deploying to Modal, run the validator:

```bash
python3 scripts/validate_workflow.py workflows/sora-removal-production.json
```

The check ensures the workflow still contains the SAM2, DiffuEraser, and
SaveImage nodes that the automation relies on.

> The CLI will refuse to run until the JSON is present, preventing accidental
> launches with an invalid workflow configuration.
