// Kontext-Style main functionality
document.addEventListener('DOMContentLoaded', function() {
    // 移除了模态框功能，现在只保留图片导航
    
    // 图片导航功能
    function initImageNavigation() {
        document.querySelectorAll('.style-display').forEach(display => {
            const container = display.querySelector('.style-container');
            const images = Array.from(container.querySelectorAll('.style-image'));
            const prevBtn = display.querySelector('.prev-btn');
            const nextBtn = display.querySelector('.next-btn');
            const currentSlideSpan = display.querySelector('.current-slide');
            const totalSlidesSpan = display.querySelector('.total-slides');
            
            if (images.length === 0) return;
            
            let currentIndex = 0;
            
            function showImage(index) {
                images.forEach(img => img.classList.remove('active'));
                images[index].classList.add('active');
                
                currentSlideSpan.textContent = index + 1;
                totalSlidesSpan.textContent = images.length;
                
                prevBtn.disabled = index === 0;
                nextBtn.disabled = index === images.length - 1;
                
                currentIndex = index;
            }
            
            prevBtn.addEventListener('click', () => {
                if (currentIndex > 0) {
                    showImage(currentIndex - 1);
                }
            });
            
            nextBtn.addEventListener('click', () => {
                if (currentIndex < images.length - 1) {
                    showImage(currentIndex + 1);
                }
            });
            
            // 键盘导航
            display.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft' && currentIndex > 0) prevBtn.click();
                if (e.key === 'ArrowRight' && currentIndex < images.length - 1) nextBtn.click();
            });
            
            // 初始化显示
            showImage(0);
        });
    }

    // 当前版本未使用自动画廊
    
    // 初始化
    initImageNavigation();
    // initAutoGallery 已禁用

    function initGalleryModal() {
        const modal = document.getElementById('gallery-modal');
        const title = document.getElementById('gallery-modal-title');
        const comparison = document.getElementById('gallery-comparison');
        const slider = document.getElementById('gallery-slider');
        const beforeImg = document.getElementById('gallery-before');
        const afterImg = document.getElementById('gallery-after');
        const afterTag = document.getElementById('gallery-after-tag');
        const toggles = Array.from(modal.querySelectorAll('.gallery-toggle'));

        if (!modal || !comparison || !slider || !beforeImg || !afterImg) return;

        let isDragging = false;
        let currentExample = null;
        let currentAfterKey = 'base';

        function setAfterKey(key) {
            currentAfterKey = key;
            toggles.forEach(btn => {
                const active = btn.dataset.after === key;
                btn.classList.toggle('active', active);
                btn.setAttribute('aria-selected', active ? 'true' : 'false');
            });

            if (!currentExample) return;
            const src = currentExample.dataset[key];
            afterImg.src = src;
            afterImg.alt = key === 'nft' ? 'LucidFlux(+LucidNFT) output' : 'LucidFlux output';
            afterTag.textContent = key === 'nft' ? 'LucidFlux(+LucidNFT)' : 'LucidFlux';
        }

        function updateSlider(x) {
            const rect = comparison.getBoundingClientRect();
            const percentage = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
            slider.style.left = percentage + '%';
            afterImg.style.clipPath = `polygon(${percentage}% 0%, 100% 0%, 100% 100%, ${percentage}% 100%)`;
        }

        function openModal(exampleBtn) {
            currentExample = exampleBtn;
            const id = exampleBtn.dataset.example || '';
            title.textContent = `Example ${id}`;

            beforeImg.src = exampleBtn.dataset.lq;
            beforeImg.alt = `Example ${id} LQ input`;
            setAfterKey('base');

            modal.classList.add('is-open');
            modal.setAttribute('aria-hidden', 'false');
            document.body.style.overflow = 'hidden';

            // Center slider after layout.
            requestAnimationFrame(() => updateSlider(comparison.getBoundingClientRect().left + comparison.getBoundingClientRect().width * 0.5));
        }

        function closeModal() {
            modal.classList.remove('is-open');
            modal.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = '';
            isDragging = false;
        }

        document.querySelectorAll('.lq-card').forEach(btn => {
            btn.addEventListener('click', () => openModal(btn));
        });

        modal.querySelectorAll('[data-gallery-close]').forEach(el => {
            el.addEventListener('click', closeModal);
        });

        toggles.forEach(btn => {
            btn.addEventListener('click', () => setAfterKey(btn.dataset.after));
        });

        slider.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });
        slider.addEventListener('touchstart', (e) => {
            isDragging = true;
            e.preventDefault();
        }, { passive: false });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            updateSlider(e.clientX);
        });
        document.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            updateSlider(e.touches[0].clientX);
        }, { passive: true });

        document.addEventListener('mouseup', () => { isDragging = false; });
        document.addEventListener('touchend', () => { isDragging = false; });

        comparison.addEventListener('click', (e) => {
            if (e.target === slider || e.target.closest('.slider-handle')) return;
            updateSlider(e.clientX);
        });

        document.addEventListener('keydown', (e) => {
            if (!modal.classList.contains('is-open')) return;
            if (e.key === 'Escape') closeModal();
            if (e.key === 'ArrowLeft') setAfterKey('base');
            if (e.key === 'ArrowRight') setAfterKey('nft');
        });
    }

    initGalleryModal();
});
