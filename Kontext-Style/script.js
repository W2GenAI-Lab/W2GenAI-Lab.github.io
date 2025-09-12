// Kontext-Style main functionality
document.addEventListener('DOMContentLoaded', function() {
    // 模态框功能
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');
    const captionText = document.getElementById('caption');
    const resolutionText = document.getElementById('resolution');
    const closeBtn = document.querySelector('.close');
    const prevBtn = document.querySelector('.prev');
    const nextBtn = document.querySelector('.next');
    const thumbnails = document.querySelector('.thumbnails');
    
    let currentImageIndex = 0;
    let galleryImages = [];
    
    // 收集所有图片
    function collectGalleryImages() {
        galleryImages = Array.from(document.querySelectorAll('.style-image, .overview-image'));
        generateThumbnails();
    }
    
    // 生成缩略图
    function generateThumbnails() {
        thumbnails.innerHTML = '';
        galleryImages.forEach((img, index) => {
            const thumb = document.createElement('img');
            thumb.src = img.src;
            thumb.alt = img.alt;
            thumb.addEventListener('click', () => {
                currentImageIndex = index;
                showModal(img);
            });
            thumbnails.appendChild(thumb);
        });
    }
    
    // 显示模态框
    function showModal(imgElement) {
        modal.style.display = 'block';
        modalImg.style.backgroundImage = `url(${imgElement.src})`;
        captionText.textContent = imgElement.alt;
        resolutionText.textContent = '';
        
        // 更新缩略图状态
        const thumbs = thumbnails.querySelectorAll('img');
        thumbs.forEach((thumb, index) => {
            thumb.classList.toggle('thumbnail-active', index === currentImageIndex);
        });
    }
    
    // 关闭模态框
    function closeModal() {
        modal.style.display = 'none';
    }
    
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
            
            // 图片点击事件（模态框）
            images.forEach((img, index) => {
                img.addEventListener('click', () => {
                    currentImageIndex = galleryImages.indexOf(img);
                    showModal(img);
                });
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
    
    // 事件监听
    document.querySelectorAll('.style-image, .overview-image').forEach(img => {
        img.addEventListener('click', function() {
            currentImageIndex = galleryImages.indexOf(this);
            showModal(this);
        });
    });
    
    closeBtn.addEventListener('click', closeModal);
    
    prevBtn.addEventListener('click', () => {
        currentImageIndex = (currentImageIndex - 1 + galleryImages.length) % galleryImages.length;
        showModal(galleryImages[currentImageIndex]);
    });
    
    nextBtn.addEventListener('click', () => {
        currentImageIndex = (currentImageIndex + 1) % galleryImages.length;
        showModal(galleryImages[currentImageIndex]);
    });
    
    // 点击模态框背景关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // 键盘导航
    document.addEventListener('keydown', (e) => {
        if (modal.style.display === 'block') {
            if (e.key === 'Escape') closeModal();
            if (e.key === 'ArrowLeft') prevBtn.click();
            if (e.key === 'ArrowRight') nextBtn.click();
        }
    });
    
    // 初始化
    collectGalleryImages();
    initImageNavigation();
});