using UnityEngine;

/// <summary>
/// Bridges a SessionThumbnailPanel's decoded texture to a MeshRenderer
/// in the runtime-generated panel hierarchy (used when no RawImage is wired).
/// Separated into its own file as Unity requires each MonoBehaviour
/// to live in a file matching its class name.
/// </summary>
public class ThumbnailRendererBridge : MonoBehaviour
{
    public Renderer              thumbnailRenderer;
    public SessionThumbnailPanel panel;
    [Tooltip("Flip only this panel's texture horizontally, via _MainTex scale/offset. " +
             "NOTE: this only works on shaders that run TRANSFORM_TEX on _MainTex_ST — " +
             "Custom/UnlitDoubleSided does, but Sprites/Default ignores tiling/offset " +
             "entirely, so on a fallback shader this silently does nothing. The RGB " +
             "diamond panels therefore correct their mirror in geometry instead (see " +
             "InterventionRgbPanelSpawner.SpawnPanel) and leave this false; the selector " +
             "grid needs no correction at all. Left in place for callers that know their " +
             "shader honours it.")]
    public bool                   flipHorizontal;
    [Tooltip("Flip only this panel's texture vertically.")]
    public bool                   flipVertical;

    private Texture _lastTex;

    void Update()
    {
        if (panel == null || thumbnailRenderer == null) return;
        var tex = panel.LastTexture;
        if (tex == null) return;

        var mat = thumbnailRenderer.material;
        Vector2 textureScale = new Vector2(flipHorizontal ? -1f : 1f,
                                           flipVertical ? -1f : 1f);
        Vector2 textureOffset = new Vector2(flipHorizontal ? 1f : 0f,
                                            flipVertical ? 1f : 0f);

        // Re-apply whenever the flip is not actually in effect, not only when the
        // texture OBJECT changes.
        //
        // SessionThumbnailPanel allocates ONE Texture2D per panel and LoadImage()s into
        // it every frame (`if (_tex == null) _tex = new Texture2D(...)`), so
        // `tex != _lastTex` fires exactly once, on the first decoded frame. That was
        // enough only as long as nothing else ever touched the material afterwards --
        // any later reassignment (a no-signal colour reset, a material swap) silently
        // dropped the horizontal flip with no path back, and the panel stayed mirrored
        // for the rest of the session with no error anywhere. Comparing against the
        // material's live state is O(1) per frame and cannot get stuck.
        // Only interrogate the material when a flip is actually requested. With no flip
        // there is nothing that can be dropped, and a shader without a _MainTex property
        // would otherwise report a mismatch forever and re-upload every frame.
        bool flipRequested = flipHorizontal || flipVertical;
        bool flipMissing = flipRequested && mat.GetTextureScale("_MainTex") != textureScale;
        if (tex == _lastTex && !flipMissing) return;

        mat.SetTexture("_BaseMap", tex);
        mat.SetTexture("_MainTex", tex);
        mat.SetTextureScale("_BaseMap", textureScale);
        mat.SetTextureOffset("_BaseMap", textureOffset);
        mat.SetTextureScale("_MainTex", textureScale);
        mat.SetTextureOffset("_MainTex", textureOffset);
        // Reset color to white so the texture displays at full brightness.
        // The no-signal grey would otherwise darken the image to ~45%.
        mat.SetColor("_Color",     Color.white);
        mat.SetColor("_BaseColor", Color.white);
        _lastTex = tex;
    }
}
