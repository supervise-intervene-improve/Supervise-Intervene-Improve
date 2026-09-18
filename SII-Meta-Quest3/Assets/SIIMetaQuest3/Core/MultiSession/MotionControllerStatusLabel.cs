using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Minimal head-relative status label for SIIScene_MotionControllerOnly.
/// Shows the current publisher state and port so the operator knows
/// the ZMQ stream is live without needing a logcat connection.
///
/// Attach to any GameObject in the standalone MC-only scene.
/// The label is parented to CenterEyeAnchor and rendered in front of the user.
/// </summary>
public class MotionControllerStatusLabel : MonoBehaviour
{
    [Header("Publisher Reference")]
    public MotionControllerZmqPublisher publisher;

    [Header("HUD Position")]
    public Vector3 localPosition = new Vector3(0f, 0.05f, 0.85f);
    public string preferredAnchorName = "CenterEyeAnchor";

    [Header("Style")]
    public int fontSize = 46;
    public float characterSize = 0.008f;
    public Color activeColor   = new Color(0f, 0.9f, 1f, 1f);   // cyan
    public Color waitingColor  = new Color(1f, 0.6f, 0f, 1f);   // amber

    private Transform _anchor;
    private GameObject _label;
    private TextMesh _text;
    private bool _wasPublishing;

    void Start()
    {
        _anchor = ResolveAnchor();
        CreateLabel();
        UpdateLabel();
    }

    void Update()
    {
        if (_anchor == null) _anchor = ResolveAnchor();
        if (_anchor != null && _label != null)
        {
            _label.transform.SetParent(_anchor, false);
            _label.transform.localPosition = localPosition;
            _label.transform.localRotation = Quaternion.identity;
        }

        bool nowPublishing = publisher != null && publisher.IsPublishing;
        if (nowPublishing != _wasPublishing)
        {
            _wasPublishing = nowPublishing;
            UpdateLabel();
        }
    }

    private void UpdateLabel()
    {
        if (_text == null) return;
        if (publisher == null)
        {
            _text.text  = "MC publisher not assigned";
            _text.color = waitingColor;
            return;
        }
        if (publisher.IsPublishing)
        {
            _text.text  = $"MC streaming :{publisher.zmqPort}";
            _text.color = activeColor;
        }
        else
        {
            _text.text  = $"MC waiting… :{publisher.zmqPort}";
            _text.color = waitingColor;
        }
    }

    private Transform ResolveAnchor()
    {
        var go = GameObject.Find(preferredAnchorName);
        if (go != null) return go.transform;
        return Camera.main != null ? Camera.main.transform : null;
    }

    private void CreateLabel()
    {
        if (_anchor == null) return;
        _label = new GameObject("MCStatusLabel");
        _label.transform.SetParent(_anchor, false);

        _text = _label.AddComponent<TextMesh>();
        var font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (font != null) _text.font = font;
        _text.fontSize     = fontSize;
        _text.characterSize = characterSize;
        _text.anchor       = TextAnchor.MiddleCenter;
        _text.alignment    = TextAlignment.Center;
        _text.fontStyle    = FontStyle.Bold;
        _text.color        = waitingColor;

        var mr = _label.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            if (font != null) mr.sharedMaterial = font.material;
            mr.shadowCastingMode    = ShadowCastingMode.Off;
            mr.receiveShadows       = false;
            mr.lightProbeUsage      = LightProbeUsage.Off;
            mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }
    }
}
