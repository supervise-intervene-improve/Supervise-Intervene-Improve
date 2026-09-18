using UnityEngine;

/// <summary>
/// Spawns an empty GameObject named "MujocoScene" in front of the headset so
/// components that look up the scene anchor by name (e.g.
/// GpuMergedPointCloudLoader.ResolveSceneRoot) find something to parent
/// under, even when MujocoPublisher is disabled (--no_mujoco_publisher).
///
/// Used by InterventionSessionBootstrap when the user enters InterventionV1
/// from the multi-session selector — no real scene mesh is published, but we
/// still want the point cloud to render at a useful location instead of at
/// world origin.
/// </summary>
public class SceneAnchorStub : MonoBehaviour
{
    [Header("Anchor")]
    public string anchorName = "MujocoScene";

    [Header("Placement (camera-local at spawn time)")]
    [Tooltip("X = right, Y = up, Z = forward (from headset).")]
    public Vector3 forwardOffset = new Vector3(0f, -0.3f, 1.0f);

    [Tooltip("If an existing GameObject with anchorName is found, re-position it. "
           + "If false, leave the existing one alone.")]
    public bool reparentExistingIfFound = false;

    void Start()
    {
        try
        {
            var existing = GameObject.Find(anchorName);
            if (existing != null && !reparentExistingIfFound)
            {
                Debug.Log($"[SceneAnchorStub] '{anchorName}' already exists at {existing.transform.position} — leaving alone.");
                return;
            }

            var cam = Camera.main;
            Vector3 pos;
            if (cam != null)
            {
                pos = cam.transform.position
                    + cam.transform.forward * forwardOffset.z
                    + cam.transform.up      * forwardOffset.y
                    + cam.transform.right   * forwardOffset.x;
            }
            else
            {
                pos = Vector3.zero;
                Debug.LogWarning("[SceneAnchorStub] Camera.main is null — spawning anchor at world origin.");
            }

            var go = existing != null ? existing : new GameObject(anchorName);
            go.transform.position = pos;
            go.transform.rotation = Quaternion.identity;
            Debug.Log(
                $"[SceneAnchorStub] spawned '{anchorName}' at world={pos} "
                + $"(cam={(cam != null ? cam.name : "<null>")}, offset={forwardOffset})."
            );
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[SceneAnchorStub] Start() threw: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
        }
    }
}
