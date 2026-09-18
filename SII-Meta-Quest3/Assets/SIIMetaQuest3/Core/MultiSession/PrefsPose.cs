using System.Globalization;
using UnityEngine;

/// <summary>
/// Culture-safe PlayerPrefs serialization for the three runtime-tunable poses
/// (SceneAnchorManager's MujocoScene, InterventionRgbPanelSpawner's diamond tilt,
/// RiskBarChart's chart root).
///
/// Why this exists: all three used string interpolation to write
/// "$\"{v.x},{v.y},{v.z}\"" and float.Parse() to read it back. Both use the
/// CURRENT culture. On a device whose locale uses a comma decimal separator
/// (de-DE and every other European locale), the write produces
/// "-0,21,1,15,3,46" -- SIX comma-separated tokens -- and the read's
/// "if (parts.Length != 3) return false" then rejects it. The failure is silent
/// and permanent: every restore falls back to the camera-relative default, which
/// for the point-cloud anchor is roughly eye height, i.e. the cloud comes back
/// HIGHER than where the operator left it, on every single entry.
///
/// Format() always writes InvariantCulture. TryParseVector3() reads
/// InvariantCulture, and additionally recovers the legacy 6-token comma-decimal
/// form so a pose saved by an older build is not thrown away on first upgrade.
/// </summary>
public static class PrefsPose
{
    /// <summary>Serialize a vector as three InvariantCulture floats separated by commas.</summary>
    public static string Format(Vector3 v)
    {
        return string.Format(
            CultureInfo.InvariantCulture, "{0},{1},{2}", v.x, v.y, v.z);
    }

    /// <summary>
    /// Parse a vector written by Format(), or by the legacy culture-dependent
    /// interpolation. Returns false (leaving <paramref name="v"/> at zero) when the
    /// string is neither.
    /// </summary>
    public static bool TryParseVector3(string s, out Vector3 v)
    {
        v = Vector3.zero;
        if (string.IsNullOrEmpty(s)) return false;

        var parts = s.Split(',');

        // Normal case: three InvariantCulture tokens.
        if (parts.Length == 3)
            return TryParse3(parts[0], parts[1], parts[2], CultureInfo.InvariantCulture, out v);

        // Legacy case: a comma-decimal locale wrote each float as "a,b", so the three values
        // arrived as six tokens. Re-pair them and parse with a comma-decimal culture, which
        // recovers the operator's saved pose instead of discarding it on the upgrade build.
        //
        // Only the all-three-had-decimals form (exactly 6 tokens) is recoverable. A value that
        // happened to be integral prints with no separator at all, so 4- and 5-token strings are
        // genuinely ambiguous about WHICH value was split and are rejected -- the caller falls
        // back and one re-save writes a clean InvariantCulture string. Guessing here would risk
        // restoring a plausible-looking but wrong pose, which is worse than falling back.
        if (parts.Length == 6)
        {
            var commaDecimal = (CultureInfo)CultureInfo.InvariantCulture.Clone();
            commaDecimal.NumberFormat.NumberDecimalSeparator = ",";
            return TryParse3(
                parts[0] + "," + parts[1],
                parts[2] + "," + parts[3],
                parts[4] + "," + parts[5],
                commaDecimal, out v);
        }

        return false;
    }

    private static bool TryParse3(string a, string b, string c, CultureInfo culture, out Vector3 v)
    {
        v = Vector3.zero;
        const NumberStyles style = NumberStyles.Float;
        if (!float.TryParse(a, style, culture, out float x)) return false;
        if (!float.TryParse(b, style, culture, out float y)) return false;
        if (!float.TryParse(c, style, culture, out float z)) return false;
        v = new Vector3(x, y, z);
        return true;
    }
}
