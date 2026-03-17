import { modules } from './content.js';
import { setupAnimations, clearAnimations } from './animations.js';

const navTabsContainer = document.getElementById('nav-tabs');
const appContentContainer = document.getElementById('app-content');

let currentModuleId = 'module1';

function renderNav() {
    navTabsContainer.innerHTML = '';
    Object.keys(modules).forEach(id => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${id === currentModuleId ? 'active' : ''}`;
        btn.textContent = modules[id].title;
        btn.onclick = () => switchModule(id);
        navTabsContainer.appendChild(btn);
    });
}

function renderContent() {
    clearAnimations(); // Stop current animations
    const contentHtml = modules[currentModuleId].content;
    appContentContainer.innerHTML = contentHtml;
    
    // Trigger MathJax re-render if loaded
    if (window.MathJax) {
        window.MathJax.typesetPromise([appContentContainer]).catch((err) => console.log(err.message));
    }

    // Setup module-specific animations
    setTimeout(() => {
        setupAnimations(currentModuleId);
    }, 100);
}

function switchModule(id) {
    if (currentModuleId !== id) {
        currentModuleId = id;
        renderNav();
        renderContent();
        window.scrollTo(0, 0);
    }
}

// Initialize
renderNav();
renderContent();