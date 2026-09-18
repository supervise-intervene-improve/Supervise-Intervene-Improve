using UnityEngine;

/// <summary>
/// Static store for the session selected in SIIScene_SessionSelector.
/// Survives scene loads via PlayerPrefs (no DontDestroyOnLoad needed).
/// All fields are in-memory mirrors of PlayerPrefs entries.
/// </summary>
public static class SessionRegistry
{
    private const string KeyIndex         = "MSel_SessionIndex";
    private const string KeyTopicPort     = "MSel_TopicPort";
    private const string KeyServicePort   = "MSel_ServicePort";
    private const string KeyPublisherIp   = "MSel_PublisherIp";
    private const string KeyLabel         = "MSel_Label";
    private const string KeyRgbMode       = "MSel_RgbMode";

    // -1 means "no session selected" — bootstrap is a no-op in that case.
    public static int    SelectedSessionIndex  = -1;
    public static int    SelectedTopicPort     = 7741;
    public static int    SelectedServicePort   = 7740;
    public static string SelectedPublisherIp   = "127.0.0.1";
    public static string SelectedLabel         = "";
    /// <summary>True when the launcher was started with --rgb_mode: the single view
    /// should render the diamond RGB camera panels instead of point clouds.</summary>
    public static bool   SelectedRgbMode       = false;
    /// <summary>Monotonic timestamp captured when the grid selection was committed.
    /// In-memory only: persisted selections from a previous app run start a fresh timing window.</summary>
    public static long   SelectedAtStopwatchTick = 0;

    /// <summary>Persist a session selection to PlayerPrefs and update in-memory state.</summary>
    public static void Save(int index, int topicPort, int servicePort, string publisherIp, string label, bool rgbMode = false)
    {
        SelectedSessionIndex = index;
        SelectedTopicPort    = topicPort;
        SelectedServicePort  = servicePort;
        SelectedPublisherIp  = publisherIp ?? "127.0.0.1";
        SelectedLabel        = label ?? "";
        SelectedRgbMode      = rgbMode;
        SelectedAtStopwatchTick = System.Diagnostics.Stopwatch.GetTimestamp();

        PlayerPrefs.SetInt(KeyIndex,       index);
        PlayerPrefs.SetInt(KeyTopicPort,   topicPort);
        PlayerPrefs.SetInt(KeyServicePort, servicePort);
        PlayerPrefs.SetString(KeyPublisherIp, SelectedPublisherIp);
        PlayerPrefs.SetString(KeyLabel,    SelectedLabel);
        PlayerPrefs.SetInt(KeyRgbMode,     rgbMode ? 1 : 0);
        PlayerPrefs.Save();
    }

    /// <summary>Load from PlayerPrefs into static fields. Call once at app start or before reading.</summary>
    public static void Load()
    {
        SelectedSessionIndex = PlayerPrefs.GetInt(KeyIndex, -1);
        SelectedTopicPort    = PlayerPrefs.GetInt(KeyTopicPort, 7741);
        SelectedServicePort  = PlayerPrefs.GetInt(KeyServicePort, 7740);
        SelectedPublisherIp  = PlayerPrefs.GetString(KeyPublisherIp, "127.0.0.1");
        SelectedLabel        = PlayerPrefs.GetString(KeyLabel, "");
        SelectedRgbMode      = PlayerPrefs.GetInt(KeyRgbMode, 0) != 0;
    }

    /// <summary>Clear the selection (returns to default single-session mode).</summary>
    public static void Clear()
    {
        SelectedSessionIndex = -1;
        SelectedTopicPort    = 7741;
        SelectedServicePort  = 7740;
        SelectedPublisherIp  = "127.0.0.1";
        SelectedLabel        = "";
        SelectedRgbMode      = false;
        SelectedAtStopwatchTick = 0;

        PlayerPrefs.DeleteKey(KeyIndex);
        PlayerPrefs.DeleteKey(KeyTopicPort);
        PlayerPrefs.DeleteKey(KeyServicePort);
        PlayerPrefs.DeleteKey(KeyPublisherIp);
        PlayerPrefs.DeleteKey(KeyLabel);
        PlayerPrefs.DeleteKey(KeyRgbMode);
        PlayerPrefs.Save();
    }

    public static bool HasSelection => SelectedSessionIndex >= 0;
}
