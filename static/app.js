let priceChart = null;
let currentAnalysisData = null; // Store the last API result for use in feedback

function quickSearch(ticker) {
    document.getElementById('ticker-input').value = ticker;
    analyzeStock();
}

function setLoading(loading) {
    const btn = document.getElementById('analyze-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');

    if (loading) {
        btn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnSpinner.classList.remove('hidden');
    } else {
        btn.disabled = false;
        btnText.textContent = 'Analyze Now';
        btnSpinner.classList.add('hidden');
    }
}

function formatCurrency(amount, currency = 'USD') {
    if (amount == null) return 'N/A';
    
    let symbol = '$';
    if (currency === 'INR') symbol = '₹';
    else if (currency === 'EUR') symbol = '€';
    else if (currency === 'GBP') symbol = '£';
    
    // Use the symbol and format the number
    return symbol + parseFloat(amount).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

async function analyzeStock() {
    const tickerInput = document.getElementById('ticker-input');
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) { tickerInput.focus(); return; }

    // Reset UI
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('error-box').classList.add('hidden');
    resetFeedbackUI();
    setLoading(true);

    try {
        const response = await fetch(`/api/analyze/${ticker}`);
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || 'Failed to fetch data.');

        currentAnalysisData = data;
        renderResults(data);
        document.getElementById('results-section').classList.remove('hidden');
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        document.getElementById('error-msg').textContent = `Error: ${err.message}`;
        document.getElementById('error-box').classList.remove('hidden');
    } finally {
        setLoading(false);
    }
}

function renderResults(data) {
    const currency = data.currency || 'USD';
    document.getElementById('res-ticker').textContent = data.ticker;
    document.getElementById('analysis-time').textContent = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    document.getElementById('res-price').textContent = formatCurrency(data.current_price, currency);

    // Diplomatic Banner
    const diplomaticAction = document.getElementById('res-diplomatic-action');
    diplomaticAction.textContent = data.diplomatic_signal.action;
    document.getElementById('res-diplomatic-reasoning').textContent = data.diplomatic_signal.reasoning;
    document.getElementById('res-diplomatic-score').textContent = `${data.diplomatic_signal.score}%`;

    // Technical Action Card
    const actionEl = document.getElementById('res-action');
    actionEl.textContent = data.action;
    actionEl.className = 'card-value action-value ' + data.action.toLowerCase();

    // Sentiment Card
    const sentimentEl = document.getElementById('res-sentiment-tone');
    sentimentEl.textContent = data.news_sentiment.tone;
    sentimentEl.className = 'card-value action-value ' + data.news_sentiment.tone.toLowerCase();
    document.getElementById('res-sentiment-score').textContent = data.news_sentiment.average_score;

    document.getElementById('res-time').textContent = `Optimal time: ${data.predicted_time}`;
    document.getElementById('res-success').textContent = `${data.success_rate}%`;

    const confidence = data.confidence;
    document.getElementById('res-confidence-pct').textContent = `${confidence}%`;
    document.getElementById('confidence-fill').style.width = `${confidence}%`;

    renderChart(data.chart_data, data.ticker, currency);
    renderCommunityStats(data.community, data.ticker);
    renderNews(data.news_sentiment.headlines);
}

function renderNews(headlines) {
    const listEl = document.getElementById('res-headlines');
    if (!headlines || headlines.length === 0) {
        listEl.innerHTML = '<div class="headline-placeholder">No recent news headlines found for this ticker.</div>';
        return;
    }

    listEl.innerHTML = headlines.map(h => {
        const toneClass = h.sentiment.toLowerCase();
        return `
            <div class="headline-item">
                <div class="sentiment-dot ${toneClass}"></div>
                <div class="headline-text" title="${h.title}">${h.title}</div>
                <div class="headline-score">${h.score > 0 ? '+' : ''}${h.score}</div>
            </div>
        `;
    }).join('');
}

function renderCommunityStats(community, ticker) {
    const section = document.getElementById('community-section');
    if (community && community.total > 0) {
        document.getElementById('community-ticker').textContent = ticker;
        document.getElementById('community-total').textContent = community.total;
        document.getElementById('community-correct').textContent = community.correct;
        document.getElementById('community-accuracy').textContent =
            community.accuracy !== null ? `${community.accuracy}%` : '--%';
        section.classList.remove('hidden');
    } else {
        section.classList.add('hidden');
    }
}

function resetFeedbackUI() {
    document.getElementById('feedback-buttons').classList.remove('hidden');
    document.getElementById('feedback-submitted').classList.add('hidden');
    document.getElementById('retrain-result').classList.add('hidden');
    document.getElementById('retrain-result').innerHTML = '';
    document.getElementById('retrain-btn').disabled = false;
    document.getElementById('retrain-spinner').classList.add('hidden');
    document.getElementById('retrain-icon').classList.remove('hidden');
    document.getElementById('retrain-text').textContent = 'Retrain Model with My Feedback';
}

async function submitFeedback(didItWork) {
    if (!currentAnalysisData) return;

    // Disable both buttons immediately
    document.getElementById('feedback-yes').disabled = true;
    document.getElementById('feedback-no').disabled = true;

    try {
        const payload = {
            ticker: currentAnalysisData.ticker,
            action: currentAnalysisData.action,
            confidence: currentAnalysisData.confidence,
            stop_loss: currentAnalysisData.stop_loss,
            current_price: currentAnalysisData.current_price,
            predicted_time: currentAnalysisData.predicted_time,
            did_it_work: didItWork,
            latest_features: currentAnalysisData.latest_features || {}
        };

        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // Show confirmation state
        document.getElementById('feedback-buttons').classList.add('hidden');
        document.getElementById('feedback-submitted').classList.remove('hidden');

        const thanksMsg = didItWork
            ? `✅ Great! Your success has been recorded. Total feedback: ${result.total_feedback}`
            : `📝 Thanks for telling us. We'll learn from this. Total feedback: ${result.total_feedback}`;
        document.getElementById('feedback-thanks-msg').textContent = thanksMsg;

        // Update community stats live after feedback
        if (result.global_summary) {
            // Refetch community for this ticker
            setTimeout(() => reloadCommunity(currentAnalysisData.ticker), 500);
        }

    } catch (err) {
        console.error('Feedback submission failed:', err);
        document.getElementById('feedback-buttons').classList.add('hidden');
        document.getElementById('feedback-submitted').classList.remove('hidden');
        document.getElementById('feedback-thanks-msg').textContent = 'Feedback saved locally. ✓';
    }
}

async function reloadCommunity(ticker) {
    try {
        const r = await fetch(`/api/feedback/summary`);
        const data = await r.json();
        const byTicker = data.by_ticker || {};
        if (byTicker[ticker]) {
            const stats = byTicker[ticker];
            renderCommunityStats({
                total: stats.total,
                correct: stats.correct,
                accuracy: stats.accuracy
            }, ticker);
        }
    } catch (e) {}
}

async function retrainModel() {
    if (!currentAnalysisData) return;
    const ticker = currentAnalysisData.ticker;

    const btn = document.getElementById('retrain-btn');
    const spinner = document.getElementById('retrain-spinner');
    const icon = document.getElementById('retrain-icon');
    const text = document.getElementById('retrain-text');
    const resultEl = document.getElementById('retrain-result');

    btn.disabled = true;
    spinner.classList.remove('hidden');
    icon.classList.add('hidden');
    text.textContent = 'Retraining...';
    resultEl.classList.add('hidden');

    try {
        const response = await fetch(`/api/retrain/${ticker}`, { method: 'POST' });
        const data = await response.json();

        if (!response.ok) throw new Error(data.error || 'Retrain failed.');

        spinner.classList.add('hidden');
        icon.classList.remove('hidden');
        text.textContent = 'Retrain Complete ✓';

        resultEl.innerHTML = `
            <strong>${data.message}</strong><br>
            New Success Rate: <span class="highlight">${data.new_success_rate}%</span> &nbsp;·&nbsp;
            Updated Signal: <span class="highlight">${data.updated_prediction.action}</span>
            (Confidence: <span class="highlight">${data.updated_prediction.confidence}%</span>)
        `;
        resultEl.classList.remove('hidden');

    } catch (err) {
        spinner.classList.add('hidden');
        icon.classList.remove('hidden');
        text.textContent = 'Retrain Model with My Feedback';
        btn.disabled = false;
        resultEl.innerHTML = `<span style="color: var(--accent-red)">Error: ${err.message}</span>`;
        resultEl.classList.remove('hidden');
    }
}

function renderChart(chartData, ticker, currency) {
    const ctx = document.getElementById('price-chart').getContext('2d');
    if (priceChart) priceChart.destroy();

    const symbol = currency === 'INR' ? '₹' : '$';
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(79, 172, 254, 0.25)');
    gradient.addColorStop(1, 'rgba(79, 172, 254, 0.0)');

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: `${ticker} Price`,
                data: chartData.prices,
                borderColor: 'rgba(79, 172, 254, 0.9)',
                backgroundColor: gradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(13, 18, 33, 0.95)',
                    borderColor: 'rgba(99, 179, 237, 0.2)',
                    borderWidth: 1,
                    titleColor: '#8b9ab5',
                    bodyColor: '#e8eaf6',
                    bodyFont: { size: 13, weight: '600' },
                    padding: 10,
                    callbacks: { label: (ctx) => `${symbol}${ctx.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2})}` }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#4a5568', maxTicksLimit: 8, font: { size: 11 }, maxRotation: 0 }
                },
                y: {
                    position: 'right',
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#4a5568', font: { size: 11 }, callback: (v) => `${symbol}${v.toLocaleString()}` }
                }
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const tickerInput = document.getElementById('ticker-input');
    const suggestionsList = document.getElementById('suggestions-list');

    tickerInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            analyzeStock();
            hideSuggestions();
        }
    });

    // Autocomplete Logic
    let debounceTimer;
    tickerInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const query = tickerInput.value.trim();
        
        if (query.length < 2) {
            hideSuggestions();
            return;
        }

        debounceTimer = setTimeout(() => fetchSuggestions(query), 300);
    });

    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!tickerInput.contains(e.target) && !suggestionsList.contains(e.target)) {
            hideSuggestions();
        }
    });
});

async function fetchSuggestions(query) {
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        renderSuggestions(data.results || []);
    } catch (err) {
        console.error('Search failed:', err);
    }
}

function renderSuggestions(results) {
    const listEl = document.getElementById('suggestions-list');
    if (results.length === 0) {
        hideSuggestions();
        return;
    }

    listEl.innerHTML = results.map(res => `
        <div class="suggestion-item" onclick="selectSuggestion('${res.ticker}')">
            <div class="sug-main">
                <span class="sug-ticker">${res.ticker}</span>
                <span class="sug-name">${res.name || ''}</span>
            </div>
            <span class="sug-exchange">${res.exchange}</span>
        </div>
    `).join('');
    
    listEl.classList.remove('hidden');
}

function selectSuggestion(ticker) {
    document.getElementById('ticker-input').value = ticker;
    hideSuggestions();
    analyzeStock();
}

function hideSuggestions() {
    document.getElementById('suggestions-list').classList.add('hidden');
}

