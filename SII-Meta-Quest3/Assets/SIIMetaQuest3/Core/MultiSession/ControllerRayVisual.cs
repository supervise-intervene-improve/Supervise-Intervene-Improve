using UnityEngine;

/// <summary>
/// Draws the RIGHT-controller selector pointer as a visible beam (+ a dot where it lands).
///
/// Until now the pointer was completely invisible: MultiSessionGridManager cast a ray every
/// frame and the only feedback was the hovered panel's border color, so the operator had to
/// sweep blindly to find a panel. This renders the ray they are actually pointing with —
/// by default ONLY while it is on a selectable panel (showOnlyWhenHovering), so the beam is a
/// "this panel is selectable" confirmation rather than a laser that follows the hand everywhere.
///
/// CRITICAL: this does NOT compute its own ray. It consumes the origin/direction/hit that
/// MultiSessionGridManager.UpdateRay() already produced. Meta's stock laser (OVRControllerHelper /
/// OVR Interaction ray) draws the RAW controller pose, but hover uses a low-passed direction
/// (raySmoothing, default 0.5) and a thick SphereCast (hoverSphereCastRadius, default 0.025) —
/// a raw beam would visibly disagree with which panel is actually selectable, which is worse
/// than no beam at all. Drawing the real ray keeps the visual and the hit test identical by
/// construction, and costs one 2-vertex LineRenderer with no extra casts.
///
/// Because the grid publishes that ray from its own Update(), this component follows the grid
/// automatically: in combined view (MultiSessionGridManager.enableCombinedView, the default) the
/// selector scene stays loaded while single view is active, so the beam is available for aiming
/// at the grid from inside a live session too. When the grid stops publishing (transitioning,
/// controller pose unavailable, or grid scene unloaded) the beam hides itself rather than
/// drawing a phantom pointer that cannot select anything.
///
/// Auto-added by MultiSessionGridManager — no Editor wiring needed.
/// </summary>
public class ControllerRayVisual : MonoBehaviour
{
    [Header("Geometry")]
    [Tooltip("Beam length when the ray is not hitting a panel. Set by MultiSessionGridManager "
           + "from its rayMaxDistance so the drawn beam ends exactly where the cast does.")]
    public float fallbackLength = 4.0f;
    [Tooltip("Beam width at the controller.")]
    public float widthStart = 0.006f;
    [Tooltip("Beam width at the far end. Slightly thinner than widthStart so the beam reads as "
           + "pointing away from the hand.")]
    public float widthEnd = 0.002f;
    [Tooltip("Diameter of the dot drawn where the ray lands. 0 disables the dot.")]
    public float hitDotDiameter = 0.022f;

    [Header("Visibility")]
    [Tooltip("Show the beam ONLY while the ray is on a selectable (revealed) panel. This is the "
           + "default: an always-on beam is visual noise everywhere except the grid, and in "
           + "combined view it would sweep across the point cloud / RGB panels of the live "
           + "session for no reason. Turn OFF to also draw the idleColor beam when pointing at "
           + "nothing (useful for debugging where the smoothed ray actually goes).")]
    public bool showOnlyWhenHovering = true;

    [Header("Color")]
    [Tooltip("Beam color while the ray is NOT on a selectable panel. Only ever drawn when "
           + "showOnlyWhenHovering is OFF.")]
    public Color idleColor  = new Color(0.62f, 0.78f, 1.00f, 1f);   // cool blue-white
    [Tooltip("Beam color while the ray IS on a revealed, selectable panel — the beam itself "
           + "confirms a trigger pull will select, matching the panel's hover border.")]
    public Color hoverColor = new Color(0.30f, 1.00f, 0.55f, 1f);   // green, = 'selectable'
    [Range(0.05f, 1f)]
    [Tooltip("Brightness multiplier while PC-adjust mode is active. Selection is suppressed "
           + "there (SceneAnchorManager.AdjustModeActive), so the beam dims to show it is inert "
           + "without vanishing — the operator is still aiming the anchor nudge controls.")]
    public float dimFactor = 0.35f;

    // -----------------------------------------------------------------------

    private LineRenderer _line;
    private GameObject   _dot;
    private Material     _lineMat;
    private Material     _dotMat;
    private Color        _appliedColor = new Color(-1f, -1f, -1f, -1f); // force first apply
    private bool         _visible = true;

    void Awake()
    {
        // Custom/UnlitDoubleSided is a project-local shader ASSET, so it is always in the
        // Android build. Shader.Find on a URP built-in would resolve but its variant can be
        // stripped, giving an invisible beam.
        Shader shader = Shader.Find("Custom/UnlitDoubleSided")
                     ?? Shader.Find("Sprites/Default")
                     ?? Shader.Find("Unlit/Color");

        var lineGo = new GameObject("ControllerRayVisual_Beam");
        lineGo.transform.SetParent(transform, false);
        _line = lineGo.AddComponent<LineRenderer>();
        _line.useWorldSpace   = true;      // positions come from the grid's world-space ray
        _line.positionCount   = 2;
        _line.startWidth      = widthStart;
        _line.endWidth        = widthEnd;
        _line.numCapVertices  = 0;
        _line.numCornerVertices = 0;
        _line.alignment       = LineAlignment.View;   // always faces the headset
        _line.textureMode     = LineTextureMode.Stretch;
        _line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        _line.receiveShadows    = false;
        if (shader != null)
        {
            _lineMat = new Material(shader);
            _line.material = _lineMat;
        }
        else
        {
            _lineMat = _line.material;
        }

        if (hitDotDiameter > 0f)
        {
            _dot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _dot.name = "ControllerRayVisual_HitDot";
            _dot.transform.SetParent(transform, false);
            _dot.transform.localScale = Vector3.one * hitDotDiameter;
            Destroy(_dot.GetComponent<Collider>());   // must never block the panel SphereCast
            var dotRenderer = _dot.GetComponent<Renderer>();
            if (shader != null)
            {
                _dotMat = new Material(shader);
                dotRenderer.material = _dotMat;
            }
            else
            {
                _dotMat = dotRenderer.material;
            }
            dotRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            dotRenderer.receiveShadows    = false;
        }

        SetVisible(false);   // stays hidden until the grid publishes a live ray
    }

    void OnDestroy()
    {
        if (_lineMat != null) Destroy(_lineMat);
        if (_dotMat  != null) Destroy(_dotMat);
    }

    /// <summary>LateUpdate so the beam uses the ray the grid computed in Update() THIS frame —
    /// no one-frame lag between where the beam is drawn and what is hovered.</summary>
    void LateUpdate()
    {
        // Stale ray = the grid is not pointing right now (transitioning, controller pose
        // unavailable, or no grid loaded at all). Tolerate one frame of execution-order slack.
        if (Time.frameCount - MultiSessionGridManager.RayFrame > 1)
        {
            SetVisible(false);
            return;
        }

        // During an MC intervention the right controller IS the robot arm: grid-select is
        // already ignored (see MultiSessionGridManager.UpdateRay), so a beam would imply an
        // interaction that cannot happen — and a laser sweeping the workspace is a distraction
        // while teleoperating. Selection returns on Left X (accept) / Left grip (reject).
        if (MotionControllerModeManager.McControlsLocked)
        {
            SetVisible(false);
            return;
        }

        // Hover-only (default): the beam exists to confirm "a trigger pull selects THIS panel",
        // so it appears exactly when that is true and stays out of the way otherwise.
        bool hovering = MultiSessionGridManager.IsHoveringSelectablePanel;
        if (showOnlyWhenHovering && !hovering)
        {
            SetVisible(false);
            return;
        }

        Vector3 origin    = MultiSessionGridManager.RayOrigin;
        Vector3 direction = MultiSessionGridManager.RayDirection;
        if (direction.sqrMagnitude < 0.0001f)
        {
            SetVisible(false);
            return;
        }
        direction.Normalize();

        bool    hasHit = MultiSessionGridManager.RayHasHit;
        Vector3 end    = hasHit
                       ? MultiSessionGridManager.RayHitPoint
                       : origin + direction * Mathf.Max(0.1f, fallbackLength);

        SetVisible(true);
        _line.SetPosition(0, origin);
        _line.SetPosition(1, end);

        if (_dot != null)
        {
            bool showDot = hasHit;
            if (_dot.activeSelf != showDot) _dot.SetActive(showDot);
            if (showDot)
            {
                // Pull the dot slightly back along the ray so it never z-fights with the
                // panel surface it is sitting on.
                _dot.transform.position = end - direction * (hitDotDiameter * 0.5f);
            }
        }

        Color color = hovering ? hoverColor : idleColor;
        if (SceneAnchorManager.Exists && SceneAnchorManager.AdjustModeActive)
            color = new Color(color.r * dimFactor, color.g * dimFactor, color.b * dimFactor, color.a);
        ApplyColor(color);
    }

    private void ApplyColor(Color color)
    {
        if (color == _appliedColor) return;
        _appliedColor = color;

        // The beam material is Custom/UnlitDoubleSided, which has no vertex-color input — so
        // LineRenderer.startColor/endColor would do nothing. The color must go on the material.
        // _BaseColor is set too so a URP fallback shader picks it up as well.
        if (_lineMat != null)
        {
            _lineMat.color = color;
            _lineMat.SetColor("_Color", color);
            _lineMat.SetColor("_BaseColor", color);
        }
        if (_dotMat != null)
        {
            _dotMat.color = color;
            _dotMat.SetColor("_Color", color);
            _dotMat.SetColor("_BaseColor", color);
        }
        // Kept in sync for any shader swap that DOES read vertex color.
        if (_line != null)
        {
            _line.startColor = color;
            _line.endColor   = color;
        }
    }

    private void SetVisible(bool visible)
    {
        if (_visible == visible) return;
        _visible = visible;
        if (_line != null) _line.enabled = visible;
        if (_dot  != null && !visible) _dot.SetActive(false);
    }
}
