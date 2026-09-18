using UnityEngine;
using NetMQ;

/// <summary>
/// Prevents the Unity Editor (and standalone builds) from freezing when
/// stopping play mode or quitting.
///
/// Root cause: NetMQPoller.Dispose() calls Thread.Join() internally.
/// If a subscriber's background thread is blocked waiting for a ZMQ message
/// (normal state when no publisher is connected), Join() hangs the main thread.
///
/// Fix: subscribe to Application.quitting, which fires BEFORE any OnDestroy
/// calls. NetMQConfig.Cleanup(block: false) sends a terminate signal to every
/// active ZMQ context, which makes blocked socket waits throw TerminatingException
/// and exit — allowing Join() to return quickly.
///
/// Add this component to any scene that uses NetMQ (both selector and intervention).
/// It is safe to have multiple instances; Cleanup() is idempotent.
/// </summary>
public class NetMQQuitHandler : MonoBehaviour
{
    void OnEnable()
    {
        Application.quitting += HandleQuit;
    }

    void OnDisable()
    {
        Application.quitting -= HandleQuit;
    }

    private static void HandleQuit()
    {
        try
        {
            // block:false — sends terminate signal without waiting for threads to finish.
            // The threads will exit on their own once they see the TerminatingException.
            NetMQConfig.Cleanup(false);
        }
        catch (System.Exception e)
        {
            // Swallow: Cleanup may throw if already terminated (re-entrant quit).
            Debug.LogWarning($"[NetMQQuitHandler] Cleanup error (benign): {e.Message}");
        }
    }
}
