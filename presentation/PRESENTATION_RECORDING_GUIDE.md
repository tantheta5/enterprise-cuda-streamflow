# Presentation Video Recording Guide
## GPU Specialization Capstone Project

This guide provides instructions for recording your 5–10 minute capstone demonstration and generating the public submission URL required by the Coursera peer review rubric.

---

## 1. Rubric Requirements Checklist
- **Duration**: Between **5 and 10 minutes** (aim for **7:00 to 8:00 minutes**).
  - *Rubric Note*: Videos under 5 minutes receive maximum 5/20 points ("Insufficient Content"). Videos 5 minutes or longer that clearly articulate goals, techniques, code, and next steps receive **20/20 points ("Excellent")**.
- **Content Checklist**:
  - [x] Clear articulation of enterprise goals and problem statement
  - [x] Software architecture & Google C++ Style Guide code walkthrough
  - [x] Algorithmic deep dive (Shared Memory Tiling, Constant Memory, Sorting Networks)
  - [x] Memory hierarchy & asynchronous multi-stream pipelining
  - [x] Empirical scalability benchmark results & speedup analysis
  - [x] Live terminal demonstration running `./run.sh` and CLI options
  - [x] Numerical parity verification report (zero RMSE)
  - [x] Challenges faced & technical resolutions
  - [x] Next steps & future enterprise scaling roadmap (Tensor Cores, Multi-GPU NCCL)

---

## 2. Recommended Recording Tools

### Option A: OBS Studio (Free, Open-Source, No Watermarks)
1. Install OBS Studio:
   ```bash
   sudo apt update && sudo apt install -y obs-studio
   ```
2. Set up:
   - Scene 1: Screen Capture (displays your presentation slides and terminal window).
   - Audio: Your microphone.
3. Start recording, read along with `presentation/DEMO_SCRIPT.md`, and switch between slides and your terminal.
4. Stop recording. Video will be saved as an `.mp4` file in your Videos directory.

### Option B: Loom / ScreenPal (Free Web-Based Browser Extension)
1. Visit [https://www.loom.com](https://www.loom.com).
2. Click "Record a Video" -> Select "Entire Screen" + Microphone.
3. Deliver the presentation using `presentation/DEMO_SCRIPT.md`.
4. Loom automatically hosts the video and provides an instant shareable link.

### Option C: Zoom / Microsoft Teams / Google Meet
1. Open a private meeting where you are the sole participant.
2. Share your desktop screen.
3. Click **Record to Local Computer** (or Cloud).
4. Run through `presentation/DEMO_SCRIPT.md`.
5. End meeting; the MP4 video is automatically generated.

---

## 3. Uploading & Generating the Submission URL

Once your MP4 recording is ready:
1. **YouTube** (Recommended):
   - Go to [YouTube Studio](https://studio.youtube.com).
   - Click **Create > Upload Video**.
   - Set visibility to **Unlisted** (anyone with the link can view; not publicly searchable) or **Public**.
   - Copy the video URL (e.g., `https://youtu.be/...`).
2. **Google Drive / Box / Dropbox**:
   - Upload the MP4 file to your Google Drive or Box account.
   - Right-click the file -> **Share** -> Change access to **"Anyone with the link can view"**.
   - Copy the public share link.

---

## 4. Submitting on Coursera
When submitting your Capstone Project assignment on Coursera, paste:
1. **Code Repository URL**: Your GitHub / GitLab URL containing the project repository.
2. **Project Presentation / Demonstration Video URL**: Your YouTube / Google Drive / Box video URL.
3. **Short Text Description**: The executive summary provided in `README.md`.
