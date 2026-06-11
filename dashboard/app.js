/**
 * AegisEdge Smart Surveillance System - Client Dashboard Controller
 * Author: Antigravity AI
 * Designed for real-time edge processing visualization and event investigation.
 */

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const socketStatusText = document.getElementById('socket-status');
    const pulseDot = document.querySelector('.pulse-dot');
    const systemStatusText = document.getElementById('sys-status-text');
    
    const fpsVal = document.getElementById('fps-val');
    const fpsProgress = document.getElementById('fps-progress');
    const cpuVal = document.getElementById('cpu-val');
    const cpuProgress = document.getElementById('cpu-progress');
    
    const badgeTracks = document.getElementById('badge-tracks');
    const badgeSource = document.getElementById('badge-source');
    
    const alertCountBadge = document.getElementById('alert-count');
    const alertsList = document.getElementById('alerts-list');
    const noAlertsMsg = document.getElementById('no-alerts-msg');
    const eventsLogBody = document.getElementById('events-log-body');
    
    // Interactive Controls
    const btnTriggerTest = document.getElementById('btn-trigger-test');
    const btnSnapshot = document.getElementById('btn-snapshot');
    const btnGrid = document.getElementById('btn-grid');
    const btnFullscreen = document.getElementById('btn-fullscreen');
    const videoFeedImg = document.getElementById('video-feed');
    
    // Modal Details
    const alertModal = document.getElementById('alert-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const modalScreenshot = document.getElementById('modal-screenshot');
    const modalEventType = document.getElementById('modal-event-type');
    const modalSeverity = document.getElementById('modal-severity');
    const modalConfidence = document.getElementById('modal-confidence');
    const modalDesc = document.getElementById('modal-desc');

    // Local state variables
    let totalAlerts = 0;
    let alertsCache = new Map(); // Store alert logs by ID for quick modal lookup

    // 1. Establish Socket.io Connection
    // By default, socket.io connects to the current host and port
    const socket = io({
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 10
    });

    // 2. Socket Event Listeners
    socket.on('connect', () => {
        console.log('Connected to surveillance backend socket.');
        socketStatusText.innerText = 'Connected to Edge AI';
        pulseDot.style.backgroundColor = 'var(--color-success)';
        pulseDot.style.boxShadow = '0 0 8px var(--color-success)';
        systemStatusText.className = 'status-indicator online';
        systemStatusText.innerText = 'Online';
    });

    socket.on('disconnect', () => {
        console.warn('Disconnected from surveillance backend.');
        socketStatusText.innerText = 'Reconnecting to Edge...';
        pulseDot.style.backgroundColor = 'var(--color-danger)';
        pulseDot.style.boxShadow = '0 0 8px var(--color-danger)';
        systemStatusText.className = 'status-indicator offline';
        systemStatusText.innerText = 'Offline';
    });

    // Handle System Performance Metrics (Inference frame rate, CPU use)
    socket.on('system_metrics', (metrics) => {
        if (!metrics) return;
        
        // Update FPS
        if (metrics.fps !== undefined) {
            const fpsValue = parseFloat(metrics.fps).toFixed(1);
            fpsVal.innerText = fpsValue;
            
            // Map 0-30 FPS to percentage width (cap at 100%)
            const fpsPct = Math.min((metrics.fps / 30) * 100, 100);
            fpsProgress.style.width = `${fpsPct}%`;
            
            if (metrics.fps < 10) {
                fpsProgress.className = 'progress-bar danger';
            } else if (metrics.fps < 20) {
                fpsProgress.className = 'progress-bar warning';
            } else {
                fpsProgress.className = 'progress-bar success';
            }
        }
        
        // Update CPU usage
        if (metrics.cpu_utilization !== undefined) {
            const cpuUsage = Math.round(metrics.cpu_utilization);
            cpuVal.innerText = `${cpuUsage}%`;
            cpuProgress.style.width = `${cpuUsage}%`;
            
            if (cpuUsage > 80) {
                cpuProgress.className = 'progress-bar danger';
            } else if (cpuUsage > 50) {
                cpuProgress.className = 'progress-bar warning';
            } else {
                cpuProgress.className = 'progress-bar success';
            }
        }

        // Update Tracker badge
        if (metrics.tracks_active !== undefined) {
            badgeTracks.innerText = `${metrics.tracks_active} Tracks Active`;
        }
        
        // Update Camera Source label
        if (metrics.camera_source !== undefined) {
            badgeSource.innerText = `Source: ${metrics.camera_source}`;
        }
    });

    // Handle incoming threat/incident alerts from AI pipeline
    socket.on('alert_event', (alert) => {
        console.log('Surveillance Alert Received:', alert);
        if (!alert || !alert.id) return;
        
        // Add to cache
        alertsCache.set(alert.id, alert);
        
        // Dynamic increments
        totalAlerts++;
        alertCountBadge.innerText = `${totalAlerts} New`;
        alertCountBadge.style.display = 'inline-block';
        
        // Remove empty state from alerts list if it exists
        if (noAlertsMsg) {
            noAlertsMsg.style.display = 'none';
        }
        
        // Prepend to sidebar alerts list
        createSidebarAlertCard(alert);
        
        // Add to recent event table
        addEventTableRow(alert);
    });

    // Helper: Create a gorgeous HTML card representing the threat alert
    function createSidebarAlertCard(alert) {
        const item = document.createElement('div');
        const severityClass = alert.severity === 'critical' ? 'crit' : (alert.severity === 'warning' ? 'warn' : '');
        item.className = `alert-item ${severityClass}`;
        item.dataset.alertId = alert.id;
        
        // Determine icon based on type
        let typeIcon = 'fa-circle-exclamation';
        if (alert.type.toLowerCase().includes('intrusion')) typeIcon = 'fa-draw-polygon';
        if (alert.type.toLowerCase().includes('theft') || alert.type.toLowerCase().includes('disappear')) typeIcon = 'fa-mask';
        if (alert.type.toLowerCase().includes('fight') || alert.type.toLowerCase().includes('violence')) typeIcon = 'fa-hand-fist';
        
        const timestamp = formatTime(alert.timestamp);
        
        item.innerHTML = `
            <div class="alert-header">
                <span class="alert-title-text"><i class="fa-solid ${typeIcon}"></i> ${alert.type}</span>
                <span class="alert-time">${timestamp}</span>
            </div>
            <div class="alert-body">
                ${alert.description}
            </div>
            <div class="alert-meta">
                <span>Confidence: ${(alert.confidence * 100).toFixed(0)}%</span>
                <span class="badge ${alert.severity === 'critical' ? 'badge-danger' : 'badge-warning'}">${alert.severity}</span>
            </div>
        `;
        
        // Add click listener to open the modal
        item.addEventListener('click', () => openAlertInvestigation(alert.id));
        
        // Prepend to container (newest alert on top)
        alertsList.insertBefore(item, alertsList.firstChild);
        
        // Keep list size under control (max 15 items in DOM list to prevent slowdown)
        if (alertsList.children.length > 15) {
            // Check if last element is empty message
            const lastChild = alertsList.lastChild;
            if (lastChild && lastChild !== noAlertsMsg) {
                alertsList.removeChild(lastChild);
            }
        }
    }

    // Helper: Append a detailed row to the main event tracker list
    function addEventTableRow(alert) {
        // Clear empty state row if present
        const emptyRow = eventsLogBody.querySelector('.empty-table-row');
        if (emptyRow) {
            eventsLogBody.innerHTML = '';
        }
        
        const tr = document.createElement('tr');
        const timestamp = formatTime(alert.timestamp);
        const severityBadge = alert.severity === 'critical' 
            ? `<span class="badge badge-danger">Critical</span>` 
            : `<span class="badge badge-warning">Warning</span>`;
            
        tr.innerHTML = `
            <td><strong>${timestamp}</strong></td>
            <td>${alert.type}</td>
            <td>${severityBadge}</td>
            <td>${(alert.confidence * 100).toFixed(0)}%</td>
            <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${alert.description}</td>
            <td>
                <button class="btn-view-alert" data-alert-id="${alert.id}">
                    <i class="fa-solid fa-magnifying-glass"></i> View
                </button>
            </td>
        `;
        
        // Add listener to the action button
        tr.querySelector('.btn-view-alert').addEventListener('click', (e) => {
            e.stopPropagation();
            openAlertInvestigation(alert.id);
        });
        
        // Prepend row (newest event on top)
        eventsLogBody.insertBefore(tr, eventsLogBody.firstChild);
    }

    // 3. Modal Controls & Screenshot Display
    function openAlertInvestigation(alertId) {
        const alert = alertsCache.get(alertId);
        if (!alert) return;
        
        modalEventType.innerText = alert.type;
        modalConfidence.innerText = `${(alert.confidence * 100).toFixed(1)}%`;
        modalDesc.innerText = alert.description;
        
        // Apply severity badges
        modalSeverity.innerText = alert.severity;
        modalSeverity.className = 'badge ' + (alert.severity === 'critical' ? 'badge-danger' : 'badge-warning');
        
        // Use screenshots if saved, otherwise load a visual representation using canvas or placeholder
        if (alert.screenshot_url) {
            modalScreenshot.src = alert.screenshot_url;
        } else {
            modalScreenshot.src = `https://placehold.co/640x360/1a1a2e/e94560?text=${encodeURIComponent(alert.type)}+Threat+Incident`;
        }
        
        alertModal.classList.add('active');
    }

    function closeModal() {
        alertModal.classList.remove('active');
        // Prevent flashing old image next time
        setTimeout(() => { modalScreenshot.src = ''; }, 300);
    }

    closeModalBtn.addEventListener('click', closeModal);
    
    // Close modal when clicking outside content area
    alertModal.addEventListener('click', (e) => {
        if (e.target === alertModal) {
            closeModal();
        }
    });

    // Close on Escape key press
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && alertModal.classList.contains('active')) {
            closeModal();
        }
    });

    // 4. Button & Control Operations
    btnTriggerTest.addEventListener('click', () => {
        console.log('Sending manual Test Alert trigger request to server...');
        fetch('/api/trigger_test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ type: 'Manual Test' })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Test alert triggered response:', data);
        })
        .catch(err => {
            console.error('Failed to trigger mock alert:', err);
            // Simulate locally as fallback
            const mockId = 'mock-' + Math.random().toString(36).substr(2, 9);
            const mockAlert = {
                id: mockId,
                type: "Intrusion Detection",
                severity: "critical",
                confidence: 0.89,
                description: "Mock Intrusion event simulated from Dashboard Client.",
                timestamp: new Date().toISOString(),
                screenshot_url: ""
            };
            socket.emit('test_alert_local', mockAlert); // If server supports echoing
            // Directly trigger standard alert handler just in case
            socket.listeners('alert_event')[0](mockAlert);
        });
    });

    // Take snapshot of current image feed
    btnSnapshot.addEventListener('click', () => {
        if (!videoFeedImg) return;
        const link = document.createElement('a');
        link.download = `Surveillance-Snapshot-${new Date().toISOString().replace(/:/g, '-')}.jpg`;
        link.href = videoFeedImg.src;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Toggle video styling/border guides
    btnGrid.addEventListener('click', () => {
        const hasBorder = videoFeedImg.style.border;
        if (hasBorder) {
            videoFeedImg.style.border = '';
        } else {
            videoFeedImg.style.border = '2px dashed var(--color-primary)';
        }
    });

    // Fullscreen video feed container
    btnFullscreen.addEventListener('click', () => {
        const videoContainer = document.querySelector('.video-container');
        if (!document.fullscreenElement) {
            videoContainer.requestFullscreen()
                .catch(err => {
                    alert(`Error attempting to enable full-screen mode: ${err.message}`);
                });
        } else {
            document.exitFullscreen();
        }
    });

    // 5. Utility: format timestamp beautifully
    function formatTime(isoString) {
        try {
            const date = new Date(isoString);
            if (isNaN(date.getTime())) return 'Just Now';
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        } catch (e) {
            return 'Just Now';
        }
    }
});
