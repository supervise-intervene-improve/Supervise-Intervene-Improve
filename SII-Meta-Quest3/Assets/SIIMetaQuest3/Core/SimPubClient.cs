using System;
using System.Collections.Concurrent;
using System.Text;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public class SimPubClient : MonoBehaviour
{
    [Header("SimPub connection")]
    public string simpubIp = "127.0.0.1";
    public int simpubTopicPort = 7741;

    [Header("Debug")]
    public bool logTopics = false;

    // topic -> latest payload
    private readonly ConcurrentDictionary<string, byte[]> _latest = new();

    private SubscriberSocket _sub;
    private NetMQPoller _poller;
    private volatile bool _shuttingDown;

    public bool TryGetLatest(string topic, out byte[] payload)
        => _latest.TryGetValue(topic, out payload);

    private void Start()
    {
        AsyncIO.ForceDotNet.Force();

        _sub = new SubscriberSocket();
        _sub.Options.ReceiveHighWatermark = 2;
        _sub.Options.Linger = TimeSpan.Zero;
        _sub.Connect($"tcp://{simpubIp}:{simpubTopicPort}");
        _sub.Subscribe("");

        _sub.ReceiveReady += OnMsg;

        _poller = new NetMQPoller { _sub };
        _poller.RunAsync();

        Debug.Log($"[SimPubClient] Connected tcp://{simpubIp}:{simpubTopicPort}");
    }

    private void OnDisable() => Shutdown();
    private void OnDestroy() => Shutdown();

    private void Shutdown()
    {
        if (_shuttingDown) return;
        _shuttingDown = true;

        var poller = _poller;
        var sub = _sub;
        _poller = null;
        _sub = null;

        try
        {
            if (sub != null) sub.ReceiveReady -= OnMsg;
        }
        catch { }

        try
        {
            if (poller != null && poller.IsRunning)
                poller.Stop();   // blocking stop before socket dispose (avoids poller-thread "Must not be disposed" crash)
        }
        catch { }

        try
        {
            if (sub != null)
            {
                sub.Close();
                sub.Dispose();
            }
        }
        catch { }

        try { poller?.Dispose(); } catch { }

        // IMPORTANT: no NetMQConfig.Cleanup() here (can freeze Unity Editor on Stop)
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown) return;

        string topic;
        try { topic = e.Socket.ReceiveFrameString(); }
        catch { return; }  // also swallows TerminatingException if context dies on the first frame

        byte[] payload;
        bool gotPayload;
        try { gotPayload = e.Socket.TryReceiveFrameBytes(out payload); }
        catch (TerminatingException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        catch (ObjectDisposedException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        if (!gotPayload || payload == null)
            return; // prevents blocking/hanging if publisher sends wrong framing

        _latest[topic] = payload;

        if (logTopics)
            Debug.Log($"[SimPubClient] RX topic={topic} bytes={payload.Length}");
    }
}
