using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Adds a lightweight head-relative label so the intervention scene is easy to identify in-headset.
/// </summary>
public class IntegrationV1HudLabel : MonoBehaviour
{
    [SerializeField] private string preferredAnchorName = "CenterEyeAnchor";
    [SerializeField] private string labelText = "Integration V1";
    [SerializeField] private Vector3 localPosition = new Vector3(-0.18f, 0.14f, 0.75f);
    [SerializeField] private Vector3 localEulerAngles = Vector3.zero;
    [SerializeField] private Color labelColor = new Color(1.0f, 0.83f, 0.38f, 1.0f);
    [SerializeField] private int fontSize = 40;
    [SerializeField] private float characterSize = 0.008f;

    private Transform _anchor;
    private GameObject _labelObject;

    void Awake() { enabled = false; }

    void LateUpdate()
    {
        if (_anchor == null || !_anchor.gameObject.activeInHierarchy)
            _anchor = ResolveAnchor();

        if (_anchor == null)
            return;

        if (_labelObject == null)
            CreateLabel();

        if (_labelObject == null)
            return;

        if (_labelObject.transform.parent != _anchor)
            _labelObject.transform.SetParent(_anchor, false);

        _labelObject.transform.localPosition = localPosition;
        _labelObject.transform.localRotation = Quaternion.Euler(localEulerAngles);
    }

    void OnDisable()
    {
        if (_labelObject != null)
            Destroy(_labelObject);

        _labelObject = null;
        _anchor = null;
    }

    private Transform ResolveAnchor()
    {
        var namedAnchor = GameObject.Find(preferredAnchorName);
        if (namedAnchor != null)
            return namedAnchor.transform;

        var mainCam = Camera.main;
        if (mainCam != null)
            return mainCam.transform;

        return null;
    }

    private void CreateLabel()
    {
        if (_anchor == null)
            return;

        _labelObject = new GameObject("IntegrationV1HudLabel");
        _labelObject.transform.SetParent(_anchor, false);

        var textMesh = _labelObject.AddComponent<TextMesh>();
        var font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (font != null)
            textMesh.font = font;
        textMesh.text = labelText;
        textMesh.fontSize = fontSize;
        textMesh.characterSize = characterSize;
        textMesh.anchor = TextAnchor.MiddleLeft;
        textMesh.alignment = TextAlignment.Left;
        textMesh.fontStyle = FontStyle.Bold;
        textMesh.color = labelColor;

        var meshRenderer = _labelObject.GetComponent<MeshRenderer>();
        if (meshRenderer != null)
        {
            if (font != null)
                meshRenderer.sharedMaterial = font.material;
            meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            meshRenderer.receiveShadows = false;
            meshRenderer.lightProbeUsage = LightProbeUsage.Off;
            meshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }
    }
}
