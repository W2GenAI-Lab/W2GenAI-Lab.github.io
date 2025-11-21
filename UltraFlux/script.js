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

    function initAutoGallery() {
        const carousel = document.querySelector('.gallery-showcase .gallery-carousel');
        if (!carousel) return;

        const track = carousel.querySelector('.gallery-track');
        const items = Array.from(track.querySelectorAll('.gallery-item'));
        if (items.length === 0) return;

        let current = 0;

        let slideWidth = items[0].getBoundingClientRect().width;
        const styles = getComputedStyle(track);
        const gap = parseFloat(styles.columnGap || styles.gap || 0);

        const recalc = () => {
            slideWidth = items[0].getBoundingClientRect().width;
            centerActive();
        };

        const centerActive = () => {
            if (window.innerWidth <= 900) {
                track.style.transform = '';
                return;
            }
            
            const offset = current * (slideWidth + gap);
            const adjustment = (carousel.clientWidth - slideWidth) / 2;
            track.style.transform = `translateX(-${offset - adjustment}px)`;
        };

        const updateClasses = () => {
            items.forEach(item => item.classList.remove('active', 'prev', 'next'));

            const prevIndex = (current - 1 + items.length) % items.length;
            const nextIndex = (current + 1) % items.length;

            items[current].classList.add('active');
            items[prevIndex].classList.add('prev');
            items[nextIndex].classList.add('next');

            centerActive();
        };

        updateClasses();

        const intervalId = setInterval(() => {
            current = (current + 1) % items.length;
            updateClasses();
        }, 5000);

        window.addEventListener('resize', recalc);
    }
    
    // 初始化
    initImageNavigation();
    initAutoGallery();
});
