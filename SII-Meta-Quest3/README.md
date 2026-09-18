# SII-Meta-Quest3 — Headset App

Unity project for the Meta Quest 3 app used in the VR-RGB and VR-PointCloud conditions.
The app shows the live robot cells as a selector grid in AR passthrough. It renders the
selected cell as a GPU point cloud or as four RGB camera panels, and forwards controller
input for intervention.

**Unity version:** 6000.0.24f1 — **Meta XR SDK:** 78.0.0 — **Target:** Quest 3 (Android ARM64)

---

## Scenes

| Scene | Build Index | Purpose |
|---|---|---|
| `SIIScene_SessionSelector` | 0 (startup) | 5×3 spherical grid of live session thumbnails |
| `SIIScene_InterventionV1` | 1 | Single-session point cloud view + intervention controls |

The selector is always the startup scene. The intervention scene is loaded when the operator
selects a session with the right controller trigger.

---

## Key Components

### MultiSession/
| Script | Purpose |
|---|---|
| `MultiSessionGridManager.cs` | Spawns 5×3 spherical panel grid; right-controller ray + trigger selection; UDP auto-discovery of publisher IP |
| `SessionThumbnailPanel.cs` | Per-panel NetMQ subscriber for `top/rgb`; reveal-on-first-frame (hides empty panels without killing subscriber) |
| `SessionCommandSender.cs` | Static PUSH socket cache; sends A/B/X/Y/ENTER_SINGLE/EXIT_SINGLE to session `cmd_port`; pre-warmed on hover |
| `InterventionSessionBootstrap.cs` | Runs at `DefaultExecutionOrder(-10000)` in `Awake()` — reconfigures all ZMQ subscribers to the selected session's IP/port before any `Start()` fires |
| `InterventionButtonForwarder.cs` | Polls Left X/Y, Right B, Left grip each frame; forwards as commands to `cmd_port` with 1.5s arming delay + held-release gate |
| `InterventionBackButton.cs` | Right index trigger → `EXIT_SINGLE` → return to selector (NetMQ teardown → 2-frame yield → `LoadScene`) |
| `SceneAnchorManager.cs` | Owns the `MujocoScene` GameObject the point cloud anchors to; controller-tunable pose (Left Menu = lock/unlock); PlayerPrefs persistence |
| `InterventionStatusHud.cs` | Subscribes to `SimPub/Status/intervention`; shows "Intervention has begun" HUD for ~3s when robot releases to human control. In policy mode (no `--mirror_robot`), the event is published by `PolicyStateMirror` watching `app.py`'s mode transitions (`replay→replan`), not by `runtime_impl.py`'s own state machine |
| `PublisherDiscoveryListener.cs` | Binds UDP 8720; parses the launcher's broadcast beacon; provides IP/port to `MultiSessionGridManager`; caches last beacon in statics for instant warm re-entry |
| `SessionRegistry.cs` | PlayerPrefs-backed session state (selected index, port, IP) |
| `NetMQQuitHandler.cs` | Calls `NetMQConfig.Cleanup(false)` on editor play-stop to prevent freeze |

### GPU_Merged/
| Script | Purpose |
|---|---|
| `GpuMergedPointCloudLoader.cs` | Main point cloud renderer. Merges up to 3 sources (top/right/left). GPU instanced cubes via `DrawMeshInstancedIndirect`. Render pose updated in `LateUpdate()` via `MaterialPropertyBlock`. `OverlayDepthWrite` mode writes depth for ATW stability. |
| `GpuMergedPointCloudBootstrap.cs` | Dynamically spawns and configures `GpuMergedPointCloudLoader` |
| `PointCloudLoader.cs` | Single-source legacy loader (contains shared `PointCloudRenderMode` enum) |
| `CubeRendering.shader` | `Custom/StableInstancedPoint` — GPU instanced cubes, dynamic ZWrite/ZTest via material properties, stereo-correct |

### Other
| Script | Purpose |
|---|---|
| `SimPubRgbdSubscriber.cs` | Wrist camera panel + RGBD display |
| `SimPubPointCloudSubscriber.cs` | Simple (non-GPU) point cloud receiver |
| `SimplePointCloudRenderer.cs` | CPU point cloud renderer |

---

## Point Cloud Render Mode

The `PointCloudRenderMode` enum (in `PointCloudLoader.cs`) controls ATW stability:

| Mode | ZWrite | ZTest | Use case |
|---|---|---|---|
| `DepthTestedOpaque` (0) | On | LessEqual | Opaque, fully occluding |
| `OverlayTransparent` (1) | **Off** | Always | Legacy; ATW cannot reproject → shaking on head movement |
| `OverlayDepthWrite` (2) | **On** | LessEqual | **Production default.** Same transparent appearance but writes depth so ATW reprojects correctly |

`InterventionSessionBootstrap` sets `renderMode = OverlayDepthWrite` on all loaders at scene load.

---

## Building

1. Open project in Unity Hub (version 6000.0.24f1)
2. Switch platform: **File → Build Profiles → Android → Switch Platform**
3. Build: **File → Build Profiles → Build Profiles → SII-Meta-Quest3-Release → Build**
4. Deploy: `adb install -r <apk>`

If the build fails with a Burst hash-cache error, close Unity, delete `Library/`, and reopen the project.

---

## Scene Build Order

`SIIScene_SessionSelector` **must be at build index 0** — it is the startup scene.
If `SIIScene_InterventionV1` is at index 0, the app launches into the wrong scene.
Check: **File → Build Settings → Scenes In Build**.
