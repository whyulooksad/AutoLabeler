import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Download,
  Eraser,
  FileVideo,
  Layers,
  MousePointer2,
  Play,
  Plus,
  RefreshCcw,
  Upload,
  Video,
  X,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

type SessionSummary = {
  session_id: string;
  video_name: string;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
  selected_frame: number;
  mask_names: string[];
  classes: string[];
  masks: MaskEntry[];
  frame_url: string;
  output_video_url: string | null;
  yolo_zip_url: string | null;
  video_info?: {
    original_width: number;
    original_height: number;
    processed_width: number;
    processed_height: number;
    scale_ratio: number;
  };
};

type MaskEntry = {
  name: string;
  class_id: number;
  class_name: string;
};

type PointMode = "positive" | "negative";
type TrackingPrecision = "original" | "720" | "480" | "360";

function apiUrl(path: string | null) {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? response.statusText);
  }
  return response.json();
}

function App() {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<PointMode>("positive");
  const [selectedMasks, setSelectedMasks] = useState<string[]>([]);
  const [className, setClassName] = useState("");
  const [trackingPrecision, setTrackingPrecision] = useState<TrackingPrecision>("original");
  const [endFrame, setEndFrame] = useState<number | "">("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Upload a video to begin.");
  const [health, setHealth] = useState<string>("Checking backend");

  useEffect(() => {
    requestJson<{ ok: boolean; cuda: boolean; device: string }>("/api/health")
      .then((data) => setHealth(data.cuda ? data.device : "CPU"))
      .catch((error) => setHealth(`Backend unavailable: ${error.message}`));
  }, []);

  const frameSrc = useMemo(() => apiUrl(session?.frame_url ?? null), [session]);
  const outputVideoSrc = useMemo(() => apiUrl(session?.output_video_url ?? null), [session]);
  const yoloZipSrc = useMemo(() => apiUrl(session?.yolo_zip_url ?? null), [session]);

  const run = useCallback(async <T,>(label: string, task: () => Promise<T>) => {
    setBusy(true);
    setMessage(label);
    try {
      return await task();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Operation failed");
      throw error;
    } finally {
      setBusy(false);
    }
  }, []);

  async function uploadVideo() {
    if (!file) return;
    await run("Processing video and preparing models...", async () => {
      const form = new FormData();
      form.append("video", file);
      const data = await requestJson<SessionSummary>("/api/sessions", { method: "POST", body: form });
      setSession(data);
      setSelectedMasks([]);
      setClassName("");
      setEndFrame(data.frame_count);
      setMessage("Video ready. Click on the frame to segment a target.");
    });
  }

  async function selectFrame(frameIndex: number) {
    if (!session) return;
    const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/select-frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_index: frameIndex }),
    });
    setSession(data);
    setMessage(`Selected frame ${frameIndex + 1}.`);
  }

  async function clickFrame(event: React.MouseEvent<HTMLImageElement>) {
    if (!session || !imageRef.current || busy) return;
    const rect = imageRef.current.getBoundingClientRect();
    const scaleX = session.width / rect.width;
    const scaleY = session.height / rect.height;
    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top) * scaleY);

    await run("Running SAM segmentation...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/click`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x, y, label: mode }),
      });
      setSession(data);
      setMessage(`${mode === "positive" ? "Positive" : "Negative"} point added at ${x}, ${y}.`);
    });
  }

  async function addMask() {
    if (!session) return;
    const targetClass = className.trim();
    if (!targetClass) {
      setMessage("Class name is required.");
      return;
    }
    await run("Adding mask...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/masks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_name: targetClass }),
      });
      setSession(data);
      setSelectedMasks(data.mask_names);
      setClassName(targetClass);
      setMessage(`Added ${data.mask_names[data.mask_names.length - 1]} as ${targetClass}.`);
    });
  }

  async function updateMaskClass(maskName: string, nextClassName: string) {
    if (!session) return;
    const targetClass = nextClassName.trim();
    if (!targetClass) return;
    await run("Updating mask class...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/masks/${maskName}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_name: targetClass }),
      });
      setSession(data);
      setClassName(targetClass);
      setMessage(`${maskName} class set to ${targetClass}.`);
    });
  }

  async function clearClicks() {
    if (!session) return;
    await run("Clearing clicks...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/clear-clicks`, { method: "POST" });
      setSession(data);
      setMessage("Clicks cleared.");
    });
  }

  async function clearMasks() {
    if (!session) return;
    await run("Clearing masks...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/masks`, { method: "DELETE" });
      setSession(data);
      setSelectedMasks([]);
      setMessage("Masks cleared.");
    });
  }

  async function track() {
    if (!session) return;
    await run("Tracking selected masks...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/track`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mask_names: selectedMasks,
          end_frame: endFrame === "" ? null : Number(endFrame),
          xmem_size: trackingPrecision === "original" ? null : Number(trackingPrecision),
        }),
      });
      setSession(data);
      setMessage("Tracking complete.");
    });
  }

  async function exportYolo() {
    if (!session) return;
    await run("Generating YOLO dataset...", async () => {
      const data = await requestJson<SessionSummary>(`/api/sessions/${session.session_id}/yolo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask_names: selectedMasks, end_frame: endFrame === "" ? null : Number(endFrame) }),
      });
      setSession(data);
      setMessage("YOLO dataset is ready.");
    });
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>AutoLabeler</h1>
          <p>{health}</p>
        </div>
        <div className={`status ${busy ? "busy" : ""}`}>{busy ? "Working" : message}</div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <section className="panel">
            <h2>Source</h2>
            <label className="file-picker">
              <FileVideo size={18} />
              <span>{file ? file.name : "Choose video"}</span>
              <input type="file" accept="video/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            </label>
            <button className="primary" disabled={!file || busy} onClick={uploadVideo}>
              <Upload size={18} />
              Upload
            </button>
          </section>

          <section className="panel">
            <h2>Frame</h2>
            <input
              type="range"
              min={0}
              max={Math.max(0, (session?.frame_count ?? 1) - 1)}
              value={session?.selected_frame ?? 0}
              disabled={!session || busy}
              onChange={(event) => selectFrame(Number(event.target.value))}
            />
            <div className="metrics">
              <span>{session ? `${session.selected_frame + 1} / ${session.frame_count}` : "No video"}</span>
              <span>{session ? `${Math.round(session.fps)} FPS` : ""}</span>
            </div>
            <label className="field">
              <span>End frame</span>
              <input
                type="number"
                min={1}
                max={session?.frame_count ?? 1}
                value={endFrame}
                disabled={!session || busy}
                onChange={(event) => setEndFrame(event.target.value === "" ? "" : Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Track quality</span>
              <select
                value={trackingPrecision}
                disabled={!session || busy}
                onChange={(event) => setTrackingPrecision(event.target.value as TrackingPrecision)}
              >
                <option value="original">Original</option>
                <option value="720">720</option>
                <option value="480">480</option>
                <option value="360">360</option>
              </select>
            </label>
          </section>

          <section className="panel">
            <h2>Prompt</h2>
            <div className="segmented">
              <button className={mode === "positive" ? "active" : ""} onClick={() => setMode("positive")} disabled={busy}>
                <MousePointer2 size={16} />
                Positive
              </button>
              <button className={mode === "negative" ? "active" : ""} onClick={() => setMode("negative")} disabled={busy}>
                <X size={16} />
                Negative
              </button>
            </div>
            <label className="field">
              <span>Class</span>
              <input
                list="class-options"
                value={className}
                disabled={!session || busy}
                onChange={(event) => setClassName(event.target.value)}
                placeholder="class name"
              />
            </label>
            <datalist id="class-options">
              {(session?.classes ?? []).map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
            <div className="toolbar">
              <button disabled={!session || busy} onClick={addMask}>
                <Plus size={18} />
                Add
              </button>
              <button disabled={!session || busy} onClick={clearClicks}>
                <Eraser size={18} />
                Clear
              </button>
              <button disabled={!session || busy} onClick={clearMasks}>
                <RefreshCcw size={18} />
                Reset
              </button>
            </div>
          </section>

          <section className="panel">
            <h2>Masks</h2>
            <div className="mask-list">
              {session?.masks.length ? (
                session.masks.map((mask) => (
                  <label key={mask.name} className="mask-row">
                    <input
                      type="checkbox"
                      checked={selectedMasks.includes(mask.name)}
                      onChange={(event) =>
                        setSelectedMasks((current) =>
                          event.target.checked ? [...current, mask.name] : current.filter((item) => item !== mask.name),
                        )
                      }
                    />
                    <span>{mask.name}</span>
                    <input
                      list="class-options"
                      defaultValue={mask.class_name}
                      disabled={busy}
                      onBlur={(event) => updateMaskClass(mask.name, event.currentTarget.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.currentTarget.blur();
                        }
                      }}
                    />
                  </label>
                ))
              ) : (
                <p className="empty">No masks yet.</p>
              )}
            </div>
            <button className="primary" disabled={!session || busy} onClick={track}>
              <Play size={18} />
              Track
            </button>
            <button disabled={!session || busy} onClick={exportYolo}>
              <Layers size={18} />
              YOLO
            </button>
          </section>
        </aside>

        <section className="stage">
          <div className="image-surface">
            {frameSrc ? (
              <img ref={imageRef} src={frameSrc} onClick={clickFrame} />
            ) : (
              <div className="placeholder">
                <Video size={42} />
              </div>
            )}
          </div>
          <div className="output-strip">
            {outputVideoSrc && (
              <a href={outputVideoSrc} target="_blank" rel="noreferrer">
                <Video size={18} />
                Result video
              </a>
            )}
            {yoloZipSrc && (
              <a href={yoloZipSrc}>
                <Download size={18} />
                YOLO zip
              </a>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
