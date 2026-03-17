let animationFrames = [];

export function clearAnimations() {
    animationFrames.forEach(id => cancelAnimationFrame(id));
    animationFrames = [];
}

export function setupAnimations(moduleId) {
    if (moduleId === 'module1') {
        initRegression();
        initNeuralNet();
    } else if (moduleId === 'module3') {
        initKMeans();
        initQLearning();
    } else if (moduleId === 'module4') {
        initAttention();
        initSoftmax();
    }
}

// ==========================================
// MODULE 1: Regression
// ==========================================
function initRegression() {
    const canvas = document.getElementById('regression-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    let points = [];
    let m = 0; // slope
    let b = canvas.height / 2; // intercept
    let learningRate = 0.00005;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        for(let i=0; i<canvas.width; i+=40) { ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i,canvas.height); ctx.stroke(); }
        for(let i=0; i<canvas.height; i+=40) { ctx.beginPath(); ctx.moveTo(0,i); ctx.lineTo(canvas.width,i); ctx.stroke(); }

        // Gradient Descent Step
        if (points.length > 1) {
            let m_grad = 0;
            let b_grad = 0;
            let total_error = 0;

            for (let i = 0; i < points.length; i++) {
                let x = points[i].x;
                let y = points[i].y;
                let guess = m * x + b;
                let error = y - guess;
                total_error += error * error;
                
                m_grad += - (2/points.length) * x * error;
                b_grad += - (2/points.length) * error;
            }

            m -= m_grad * learningRate;
            b -= b_grad * (learningRate * 10000); 

            document.getElementById('reg-mse').innerText = (total_error / points.length).toFixed(2);
            document.getElementById('reg-m').innerText = m.toFixed(2);
            document.getElementById('reg-b').innerText = b.toFixed(2);
        }

        // Draw Line
        ctx.beginPath();
        ctx.moveTo(0, b);
        ctx.lineTo(canvas.width, m * canvas.width + b);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Draw points
        ctx.fillStyle = '#1e40af';
        points.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
            ctx.fill();
        });

        animationFrames.push(requestAnimationFrame(draw));
    }

    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        points.push({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        });
    });

    const clearBtn = document.getElementById('btn-reg-clear');
    if (clearBtn) {
        clearBtn.onclick = () => {
            points = [];
            m = 0;
            b = canvas.height / 2;
            document.getElementById('reg-mse').innerText = '0.00';
            document.getElementById('reg-m').innerText = '0.00';
            document.getElementById('reg-b').innerText = '0.00';
        };
    }

    draw();
}

// ==========================================
// MODULE 1: Neural Network Animation
// ==========================================
function initNeuralNet() {
    const canvas = document.getElementById('nn-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const layers = [4, 6, 6, 2];
    const nodes = [];
    const nodeRadius = 8;
    const padding = 40;
    
    let time = 0;

    // Initialize nodes
    layers.forEach((count, layerIndex) => {
        const x = padding + (layerIndex * (canvas.width - 2 * padding) / (layers.length - 1));
        for (let i = 0; i < count; i++) {
            const y = padding + (i * (canvas.height - 2 * padding) / (count === 1 ? 1 : count - 1));
            const yOffset = count === 1 ? (canvas.height)/2 : y;
            nodes.push({ layer: layerIndex, x, y: yOffset, activation: Math.random() });
        }
    });

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        time += 0.05;

        // Draw connections
        ctx.lineWidth = 1;
        nodes.forEach(n1 => {
            nodes.forEach(n2 => {
                if (n2.layer === n1.layer + 1) {
                    const signal = (Math.sin(time - n1.x/50) + 1) / 2;
                    ctx.strokeStyle = `rgba(56, 189, 248, ${signal * 0.4})`;
                    ctx.beginPath();
                    ctx.moveTo(n1.x, n1.y);
                    ctx.lineTo(n2.x, n2.y);
                    ctx.stroke();
                }
            });
        });

        // Draw nodes
        nodes.forEach(n => {
            n.activation = (Math.sin(time * 2 + n.y) + 1) / 2;
            
            ctx.beginPath();
            ctx.arc(n.x, n.y, nodeRadius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(96, 165, 250, ${n.activation})`;
            ctx.fill();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Glow effect
            if(n.activation > 0.8) {
                ctx.shadowBlur = 15;
                ctx.shadowColor = '#38bdf8';
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        });

        animationFrames.push(requestAnimationFrame(draw));
    }
    draw();
}

// ==========================================
// MODULE 3: K-Means
// ==========================================
function initKMeans() {
    const canvas = document.getElementById('kmeans-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    let points = [];
    let centroids = [];
    const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b'];
    const k = 3;

    function generateData() {
        points = [];
        centroids = [];
        
        for(let i=0; i<k; i++) {
            const cx = Math.random() * (canvas.width - 100) + 50;
            const cy = Math.random() * (canvas.height - 100) + 50;
            for(let j=0; j<50; j++) {
                points.push({
                    x: cx + (Math.random() * 80 - 40),
                    y: cy + (Math.random() * 80 - 40),
                    cluster: -1
                });
            }
        }
        
        for(let i=0; i<k; i++) {
            centroids.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                color: colors[i]
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        points.forEach(p => {
            ctx.fillStyle = p.cluster === -1 ? '#94a3b8' : colors[p.cluster];
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            ctx.fill();
        });

        centroids.forEach((c) => {
            ctx.fillStyle = c.color;
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.rect(c.x - 8, c.y - 8, 16, 16);
            ctx.fill();
            ctx.stroke();
        });
    }

    function step() {
        // Expectation: Assign points
        points.forEach(p => {
            let minDist = Infinity;
            let cluster = -1;
            centroids.forEach((c, idx) => {
                const dist = Math.hypot(p.x - c.x, p.y - c.y);
                if (dist < minDist) {
                    minDist = dist;
                    cluster = idx;
                }
            });
            p.cluster = cluster;
        });

        draw(); 

        // Maximization: Move centroids
        setTimeout(() => {
            centroids.forEach((c, idx) => {
                const clusterPoints = points.filter(p => p.cluster === idx);
                if (clusterPoints.length > 0) {
                    const sumX = clusterPoints.reduce((sum, p) => sum + p.x, 0);
                    const sumY = clusterPoints.reduce((sum, p) => sum + p.y, 0);
                    c.x = sumX / clusterPoints.length;
                    c.y = sumY / clusterPoints.length;
                }
            });
            draw();
        }, 400);
    }

    const btnStep = document.getElementById('btn-kmeans-step');
    const btnReset = document.getElementById('btn-kmeans-reset');
    if (btnStep) btnStep.onclick = step;
    if (btnReset) btnReset.onclick = () => { generateData(); draw(); };

    generateData();
    draw();
}

// ==========================================
// MODULE 3: Q-Learning Grid
// ==========================================
function initQLearning() {
    const canvas = document.getElementById('qlearning-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const gridSize = 4;
    const cellSize = canvas.width / gridSize;
    
    let qTable = Array(gridSize).fill().map(() => Array(gridSize).fill().map(() => [0,0,0,0]));
    
    let agent = {x: 0, y: 0};
    const goal = {x: 3, y: 3};
    const pit = {x: 2, y: 1};

    function drawGrid() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        for(let x=0; x<gridSize; x++) {
            for(let y=0; y<gridSize; y++) {
                const maxQ = Math.max(...qTable[x][y]);
                const intensity = Math.min(1, Math.max(0, maxQ / 100));
                
                if(x === goal.x && y === goal.y) {
                    ctx.fillStyle = '#10b981'; 
                } else if(x === pit.x && y === pit.y) {
                    ctx.fillStyle = '#ef4444'; 
                } else {
                    ctx.fillStyle = `rgba(16, 185, 129, ${intensity})`; 
                    if (maxQ === 0) ctx.fillStyle = '#f8fafc';
                }
                
                ctx.fillRect(x*cellSize, y*cellSize, cellSize, cellSize);
                ctx.strokeStyle = '#cbd5e1';
                ctx.strokeRect(x*cellSize, y*cellSize, cellSize, cellSize);

                if(x === agent.x && y === agent.y) {
                    ctx.fillStyle = '#3b82f6';
                    ctx.beginPath();
                    ctx.arc(x*cellSize + cellSize/2, y*cellSize + cellSize/2, cellSize/3, 0, Math.PI*2);
                    ctx.fill();
                }
            }
        }
    }

    function trainEpochs(epochs) {
        const alpha = 0.1;
        const gamma = 0.9;
        
        for(let i=0; i<epochs; i++) {
            let currX = 0, currY = 0;
            let steps = 0;
            
            while(steps < 30) {
                let action;
                if(Math.random() < 0.2) {
                    action = Math.floor(Math.random() * 4);
                } else {
                    const maxQ = Math.max(...qTable[currX][currY]);
                    action = qTable[currX][currY].indexOf(maxQ);
                }

                let nextX = currX, nextY = currY;
                if(action===0) nextY = Math.max(0, currY-1);
                if(action===1) nextX = Math.min(gridSize-1, currX+1);
                if(action===2) nextY = Math.min(gridSize-1, currY+1);
                if(action===3) nextX = Math.max(0, currX-1);

                let reward = -1; 
                if(nextX===goal.x && nextY===goal.y) reward = 100;
                if(nextX===pit.x && nextY===pit.y) reward = -100;

                const maxNextQ = Math.max(...qTable[nextX][nextY]);
                qTable[currX][currY][action] = qTable[currX][currY][action] + 
                    alpha * (reward + gamma * maxNextQ - qTable[currX][currY][action]);

                currX = nextX;
                currY = nextY;
                steps++;

                if(reward === 100 || reward === -100) break;
            }
        }
        drawGrid();
    }

    let lastTime = 0;
    function animateAgent(time) {
        if(time - lastTime > 400) { 
            lastTime = time;
            
            const maxQ = Math.max(...qTable[agent.x][agent.y]);
            const action = qTable[agent.x][agent.y].indexOf(maxQ);
            
            if (maxQ > 0) { 
                if(action===0) agent.y = Math.max(0, agent.y-1);
                if(action===1) agent.x = Math.min(gridSize-1, agent.x+1);
                if(action===2) agent.y = Math.min(gridSize-1, agent.y+1);
                if(action===3) agent.x = Math.max(0, agent.x-1);
            }

            if((agent.x === goal.x && agent.y === goal.y) || (agent.x === pit.x && agent.y === pit.y)) {
                agent.x = 0; agent.y = 0; 
            }
            
            drawGrid();
        }
        animationFrames.push(requestAnimationFrame(animateAgent));
    }

    const trainBtn = document.getElementById('btn-ql-train');
    const resetBtn = document.getElementById('btn-ql-reset');
    
    if (trainBtn) trainBtn.onclick = () => trainEpochs(100);
    if (resetBtn) resetBtn.onclick = () => {
        qTable = Array(gridSize).fill().map(() => Array(gridSize).fill().map(() => [0,0,0,0]));
        agent.x = 0; agent.y = 0;
        drawGrid();
    };

    drawGrid();
    animationFrames.push(requestAnimationFrame(animateAgent));
}

// ==========================================
// MODULE 4: Attention Visualization
// ==========================================
function initAttention() {
    const canvas = document.getElementById('attention-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const words = ["The", "bank", "of", "the", "river"];
    const weights = [
        [0.8, 0.1, 0.05, 0.05, 0.0],
        [0.1, 0.4, 0.1, 0.1, 0.3], 
        [0.0, 0.1, 0.8, 0.1, 0.0],
        [0.05, 0.05, 0.0, 0.8, 0.1],
        [0.0, 0.3, 0.0, 0.1, 0.6]  
    ];

    let time = 0;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        time += 0.03;

        const padding = 60;
        const spacing = (canvas.width - padding * 2) / (words.length - 1);

        for(let i=0; i<words.length; i++) {
            for(let j=0; j<words.length; j++) {
                if (i !== j) {
                    const w = weights[i][j];
                    if (w > 0.05) {
                        const pulse = (Math.sin(time + i + j) + 1) / 2;
                        ctx.beginPath();
                        ctx.moveTo(padding + i * spacing, 50);
                        
                        const cx1 = padding + i * spacing;
                        const cy1 = 150;
                        const cx2 = padding + j * spacing;
                        const cy2 = 150;
                        
                        ctx.bezierCurveTo(cx1, cy1, cx2, cy2, padding + j * spacing, canvas.height - 50);
                        
                        ctx.strokeStyle = `rgba(79, 70, 229, ${w * 2 * pulse})`;
                        ctx.lineWidth = w * 12;
                        ctx.stroke();
                    }
                }
            }
        }

        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#1e293b';
        
        words.forEach((word, i) => {
            const x = padding + i * spacing;
            ctx.fillText(word, x, 35);
            ctx.beginPath(); ctx.arc(x, 50, 6, 0, Math.PI*2); ctx.fill();

            ctx.fillText(word, x, canvas.height - 15);
            ctx.beginPath(); ctx.arc(x, canvas.height - 50, 6, 0, Math.PI*2); ctx.fill();
        });

        animationFrames.push(requestAnimationFrame(draw));
    }

    draw();
}

// ==========================================
// MODULE 4: Softmax Calculator
// ==========================================
function initSoftmax() {
    const container = document.getElementById('softmax-sliders');
    const barsContainer = document.getElementById('softmax-bars');
    if (!container || !barsContainer) return;

    let logits = [2.0, 1.0, 0.1, -1.0];
    const labels = ['Cat', 'Dog', 'Car', 'Tree'];

    function update() {
        const exps = logits.map(Math.exp);
        const sumExp = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sumExp);

        container.innerHTML = '';
        logits.forEach((l, i) => {
            const div = document.createElement('div');
            div.className = 'flex items-center space-x-4';
            div.innerHTML = `
                <span class="w-12 font-semibold">${labels[i]}</span>
                <input type="range" min="-5" max="5" step="0.1" value="${l}" class="flex-grow accent-indigo-600">
                <span class="w-12 text-right text-gray-600 font-mono">${l.toFixed(1)}</span>
            `;
            div.querySelector('input').addEventListener('input', (e) => {
                logits[i] = parseFloat(e.target.value);
                update();
            });
            container.appendChild(div);
        });

        barsContainer.innerHTML = '';
        probs.forEach((p, i) => {
            const pct = (p * 100).toFixed(1);
            const bar = document.createElement('div');
            bar.className = 'flex flex-col items-center justify-end h-full w-16';
            bar.innerHTML = `
                <span class="text-xs mb-1 text-indigo-700 font-bold">${pct}%</span>
                <div class="w-full bg-indigo-500 rounded-t transition-all duration-200" style="height: ${p * 100}%"></div>
                <span class="mt-2 text-sm text-gray-700 font-semibold">${labels[i]}</span>
            `;
            barsContainer.appendChild(bar);
        });
    }

    update();
}