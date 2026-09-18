using System;
using System.Collections.Generic;
using NetMQ;
using NetMQ.Monitoring;
using NetMQ.Sockets;
using UnityEngine;

/// <summary>
/// Per-session ZMQ PUSH sender for forwarding A/B/X/Y button events from the
/// multi-session selector to a specific Python session's cmd_port (topic_port + 5).
/// Sockets are lazily created and cached by ip:port; sends are fire-and-forget
/// with a 50 ms timeout so the main thread never blocks on a stalled peer.
/// Each socket has a NetMQMonitor attached so connect/disconnect/retry events
/// surface in adb logcat — critical for diagnosing why a command never lands.
/// </summary>
public static class SessionCommandSender
{
    private static readonly Dictionary<string, PushSocket> _sockets = new Dictionary<string, PushSocket>();
    private static readonly Dictionary<string, NetMQMonitor> _monitors = new Dictionary<string, NetMQMonitor>();
    private static NetMQPoller _monPoller;
    private static readonly object _lock = new object();
    private static int _generation;
    private static int _sendSequence;
    private static int _monitorSequence;


    public static bool Send(string ip, int port, string cmd)
    {
        if (string.IsNullOrWhiteSpace(ip) || port <= 0 || string.IsNullOrEmpty(cmd))
        {
            Debug.LogWarning($"[SessionCommandSender] invalid send request ip='{ip}' port={port} cmd='{cmd}'");
            return false;
        }


        string key = $"{ip}:{port}";
        PushSocket sock;
        int sendId;
        int generation;
        lock (_lock)
        {
            sendId = ++_sendSequence;
            generation = _generation;
            Debug.Log(
                $"[SessionCommandSender] send begin seq={sendId} gen={generation} cmd='{cmd}' -> {key} cached={_sockets.ContainsKey(key)}"
            );

            if (!_sockets.TryGetValue(key, out sock))
            {
                try
                {
                    AsyncIO.ForceDotNet.Force();
                    sock = new PushSocket();
                    sock.Options.SendHighWatermark = 4;
                    sock.Options.Linger = TimeSpan.Zero;
                    AttachMonitor(sock, key);
                    sock.Connect($"tcp://{ip}:{port}");
                    _sockets[key] = sock;
                    Debug.Log($"[SessionCommandSender] PUSH connect seq={sendId} gen={generation} tcp://{ip}:{port}");
                }
                catch (Exception ex)
                {

                    Debug.LogWarning($"[SessionCommandSender] connect fail seq={sendId} gen={generation} {key}: {ex.Message}");
                    DropSocketLocked(key);
                    return false;
                }
            }
        }

        try
        {

            if (!sock.TrySendFrame(TimeSpan.FromMilliseconds(50), cmd))
            {
                // A timeout usually means the PUSH socket is still completing its TCP
                // connect (no pipe yet) — common on the FIRST send to a session. Do NOT
                // drop the socket: keep it warm so the next send (or the resend in the
                // grid select path) lands once connected. Dropping here is what caused
                // ENTER_SINGLE to be lost → no point clouds / no single-view mirror.
                Debug.LogWarning($"[SessionCommandSender] send timeout seq={sendId} gen={generation} {key} cmd='{cmd}' (keeping socket warm)");
                return false;
            }
            else
            {
                Debug.Log($"[SessionCommandSender] sent seq={sendId} gen={generation} '{cmd}' -> {key}");
                return true;
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[SessionCommandSender] send fail seq={sendId} gen={generation} {key} cmd='{cmd}': {ex.Message}");
            DropSocket(key);

            return false;
        }
    }

    /// <summary>Pre-create and connect the PUSH socket for ip:port WITHOUT sending, so a
    /// later Send (e.g. ENTER_SINGLE on select) finds an established pipe instead of timing
    /// out on a cold socket. Call when a panel is hovered so the connection is warm by the
    /// time the trigger fires. Safe to call repeatedly (no-op if already cached).</summary>
    public static void Warm(string ip, int port)
    {
        if (string.IsNullOrWhiteSpace(ip) || port <= 0)
            return;

        string key = $"{ip}:{port}";
        lock (_lock)
        {
            if (_sockets.ContainsKey(key))
                return;
            try
            {
                AsyncIO.ForceDotNet.Force();
                var sock = new PushSocket();
                sock.Options.SendHighWatermark = 4;
                sock.Options.Linger = TimeSpan.Zero;
                AttachMonitor(sock, key);
                sock.Connect($"tcp://{ip}:{port}");
                _sockets[key] = sock;
                Debug.Log($"[SessionCommandSender] PUSH warm-connect tcp://{ip}:{port}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[SessionCommandSender] warm fail {key}: {ex.Message}");
                DropSocketLocked(key);
            }
        }
    }

    /// <summary>Close all cached sockets. Call from scene-shutdown or quit handler.</summary>
    public static void Shutdown()
    {
        lock (_lock)
        {
            var monitors = new List<NetMQMonitor>(_monitors.Values);
            var sockets = new List<PushSocket>(_sockets.Values);
            var poller = _monPoller;
            int generation = ++_generation;

            Debug.Log(
                $"[SessionCommandSender] Shutdown gen={generation} sockets={sockets.Count} monitors={monitors.Count} pollerRunning={(poller != null && poller.IsRunning)}"
            );
            _monitors.Clear();
            _sockets.Clear();
            _monPoller = null;

            // Stop() blocks until the monitor poll loop exits before monitors/sockets below
            // are disposed — avoids "Must not be disposed" from the running poller on teardown.
            try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
            foreach (var m in monitors)
            {
                try { m.DetachFromPoller(); } catch { }
                try { m.Dispose(); } catch { }
            }
            foreach (var s in sockets)
            {
                try { s.Close(); } catch { }
                try { s.Dispose(); } catch { }
            }
            try { poller?.Dispose(); } catch { }
            Debug.Log($"[SessionCommandSender] Shutdown complete gen={generation}");
        }
    }

    private static void DropSocket(string key)
    {
        lock (_lock)
        {
            DropSocketLocked(key);
        }
    }

    private static void DropSocketLocked(string key)
    {
        if (_monitors.TryGetValue(key, out var monitor))
        {
            _monitors.Remove(key);
            try { monitor.DetachFromPoller(); } catch { }
            try { monitor.Dispose(); } catch { }
        }

        if (_sockets.TryGetValue(key, out var socket))
        {
            _sockets.Remove(key);
            try { socket.Close(); } catch { }
            try { socket.Dispose(); } catch { }
        }
    }

    private static void AttachMonitor(PushSocket sock, string key)
    {
        try
        {
            if (_monPoller == null || !_monPoller.IsRunning)
            {
                _monPoller = new NetMQPoller();
                _monPoller.RunAsync();
            }
            int monitorId = ++_monitorSequence;
            var mon = new NetMQMonitor(
                sock,
                $"inproc://mon-cmd-{monitorId}",
                SocketEvents.Connected | SocketEvents.ConnectDelayed
                | SocketEvents.ConnectRetried | SocketEvents.Disconnected);
            mon.Connected      += (_, e) => Debug.Log($"[SessionCommandSender] {key} Connected addr={e.Address}");
            mon.ConnectDelayed += (_, e) => Debug.LogWarning($"[SessionCommandSender] {key} ConnectDelayed addr={e.Address}");
            mon.ConnectRetried += (_, e) => Debug.LogWarning($"[SessionCommandSender] {key} ConnectRetried addr={e.Address}");
            mon.Disconnected   += (_, e) => Debug.LogWarning($"[SessionCommandSender] {key} Disconnected addr={e.Address}");
            mon.AttachToPoller(_monPoller);
            _monitors[key] = mon;
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[SessionCommandSender] monitor attach failed for {key}: {ex.Message}");
        }
    }
}
