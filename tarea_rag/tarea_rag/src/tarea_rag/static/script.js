const API_URL = 'http://localhost:5001/api';

// Cargar estadísticas al inicio
async function loadStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        const stats = await response.json();
        
        const statsGrid = document.getElementById('stats-grid');
        statsGrid.innerHTML = `
            <div class="stat-card">
                <h3>Clientes</h3>
                <div class="value">${stats.total_clientes}</div>
            </div>
            <div class="stat-card">
                <h3>Productos</h3>
                <div class="value">${stats.total_productos}</div>
            </div>
            <div class="stat-card">
                <h3>Ventas</h3>
                <div class="value">${stats.total_ventas}</div>
            </div>
            <div class="stat-card">
                <h3>Ingresos Totales</h3>
                <div class="value">$${stats.ingresos_totales.toFixed(2)}</div>
            </div>
        `;
    } catch (error) {
        console.error('Error cargando estadísticas:', error);
    }
}

function setQuestion(question) {
    document.getElementById('question-input').value = question;
    document.getElementById('question-input').focus();
}

function addMessage(content, isUser = false) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addLoadingMessage() {
    const messagesContainer = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loading-message';
    loadingDiv.innerHTML = `
        <div class="message-content loading">
            <span>Analizando pregunta</span>
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

function formatResult(resultado, tipo) {
    if (typeof resultado === 'number') {
        return `<div class="result-value">${resultado.toLocaleString()}</div>`;
    } else if (typeof resultado === 'string') {
        return `<div class="result-value">${resultado}</div>`;
    } else if (Array.isArray(resultado) && resultado.length > 0) {
        // Es un array de objetos (DataFrame)
        const keys = Object.keys(resultado[0]);
        let table = '<table><thead><tr>';
        keys.forEach(key => {
            table += `<th>${key}</th>`;
        });
        table += '</tr></thead><tbody>';
        resultado.slice(0, 10).forEach(row => {
            table += '<tr>';
            keys.forEach(key => {
                table += `<td>${row[key]}</td>`;
            });
            table += '</tr>';
        });
        table += '</tbody></table>';
        if (resultado.length > 10) {
            table += `<p style="margin-top: 10px; color: #666;"><em>Mostrando 10 de ${resultado.length} resultados</em></p>`;
        }
        return table;
    } else if (typeof resultado === 'object' && resultado !== null) {
        // Es un objeto/diccionario
        let content = '<div style="margin-top: 10px;">';
        Object.entries(resultado).forEach(([key, value]) => {
            content += `<div style="margin: 5px 0;"><strong>${key}:</strong> ${value}</div>`;
        });
        content += '</div>';
        return content;
    }
    return `<div class="result-value">${JSON.stringify(resultado)}</div>`;
}

async function sendQuestion() {
    const input = document.getElementById('question-input');
    const sendBtn = document.getElementById('send-btn');
    const pregunta = input.value.trim();
    
    if (!pregunta) return;
    
    // Deshabilitar input
    input.disabled = true;
    sendBtn.disabled = true;
    
    // Agregar mensaje del usuario
    addMessage(pregunta, true);
    
    // Limpiar input
    input.value = '';
    
    // Agregar mensaje de carga
    addLoadingMessage();
    
    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pregunta })
        });
        
        const data = await response.json();
        
        removeLoadingMessage();
        
        if (data.error) {
            addMessage(`<div style="color: red;">❌ Error: ${data.error}</div>`);
        } else {
            // Mostrar solo la respuesta en lenguaje natural
            let responseContent = data.respuesta || formatResult(data.resultado_raw, data.tipo);
            addMessage(responseContent);
        }
    } catch (error) {
        removeLoadingMessage();
        addMessage(`<div style="color: red;">❌ Error de conexión: ${error.message}</div>`);
    } finally {
        // Rehabilitar input
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// Cargar estadísticas al iniciar
loadStats();
