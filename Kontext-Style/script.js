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
    
    // 初始化
    initImageNavigation();
});