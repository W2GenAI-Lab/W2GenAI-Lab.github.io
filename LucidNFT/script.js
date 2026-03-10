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

    function initInlineComparisonSliders() {
        const comparisons = Array.from(document.querySelectorAll('.image-comparison'));
        if (comparisons.length === 0) return;

        const dragState = { isDragging: false, active: null };

        function updateSlider(comparisonEl, sliderEl, afterEl, x) {
            const rect = comparisonEl.getBoundingClientRect();
            const percentage = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
            sliderEl.style.left = percentage + '%';
            afterEl.style.clipPath = `polygon(${percentage}% 0%, 100% 0%, 100% 100%, ${percentage}% 100%)`;
        }

        comparisons.forEach(comparisonEl => {
            const sliderEl = comparisonEl.querySelector('.comparison-slider');
            const afterEl = comparisonEl.querySelector('.after-image');
            if (!sliderEl || !afterEl) return;

            // Drag start
            sliderEl.addEventListener('mousedown', (e) => {
                dragState.isDragging = true;
                dragState.active = { comparisonEl, sliderEl, afterEl };
                e.preventDefault();
                e.stopPropagation();
            });
            sliderEl.addEventListener('touchstart', (e) => {
                dragState.isDragging = true;
                dragState.active = { comparisonEl, sliderEl, afterEl };
                e.preventDefault();
                e.stopPropagation();
            }, { passive: false });

            // Click-to-jump
            comparisonEl.addEventListener('click', (e) => {
                if (e.target === sliderEl || e.target.closest('.slider-handle')) return;
                updateSlider(comparisonEl, sliderEl, afterEl, e.clientX);
            });
        });

        document.addEventListener('mousemove', (e) => {
            if (!dragState.isDragging || !dragState.active) return;
            updateSlider(dragState.active.comparisonEl, dragState.active.sliderEl, dragState.active.afterEl, e.clientX);
        });
        document.addEventListener('touchmove', (e) => {
            if (!dragState.isDragging || !dragState.active) return;
            updateSlider(dragState.active.comparisonEl, dragState.active.sliderEl, dragState.active.afterEl, e.touches[0].clientX);
        }, { passive: true });

        document.addEventListener('mouseup', () => { dragState.isDragging = false; dragState.active = null; });
        document.addEventListener('touchend', () => { dragState.isDragging = false; dragState.active = null; });
    }

    initInlineComparisonSliders();
});
