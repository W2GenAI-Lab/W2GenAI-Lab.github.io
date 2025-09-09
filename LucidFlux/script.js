// 图片对比滑动功能
document.addEventListener('DOMContentLoaded', function() {
    const imageComparisons = document.querySelectorAll('.image-comparison');
    
    imageComparisons.forEach(comparison => {
        const slider = comparison.querySelector('.comparison-slider');
        const afterImage = comparison.querySelector('.after-image');
        let isDragging = false;
        
        function updateSlider(x) {
            const rect = comparison.getBoundingClientRect();
            const percentage = Math.max(0, Math.min(100, ((x - rect.left) / rect.width) * 100));
            
            slider.style.left = percentage + '%';
            afterImage.style.clipPath = `polygon(${percentage}% 0%, 100% 0%, 100% 100%, ${percentage}% 100%)`;
        }
        
        function handleMouseDown(e) {
            isDragging = true;
            e.preventDefault();
        }
        
        function handleMouseMove(e) {
            if (!isDragging) return;
            updateSlider(e.clientX);
        }
        
        function handleMouseUp() {
            isDragging = false;
        }
        
        function handleTouchStart(e) {
            isDragging = true;
            e.preventDefault();
        }
        
        function handleTouchMove(e) {
            if (!isDragging) return;
            const touch = e.touches[0];
            updateSlider(touch.clientX);
        }
        
        function handleTouchEnd() {
            isDragging = false;
        }
        
        // 鼠标事件
        slider.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        
        // 触摸事件
        slider.addEventListener('touchstart', handleTouchStart);
        document.addEventListener('touchmove', handleTouchMove);
        document.addEventListener('touchend', handleTouchEnd);
        
        // 点击直接跳转
        comparison.addEventListener('click', function(e) {
            if (e.target === slider || e.target.closest('.slider-handle')) return;
            updateSlider(e.clientX);
        });
    });
    
    // 现有的模态框功能保持不变
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
        galleryImages = Array.from(document.querySelectorAll('.gallery-image, .before-image, .after-image'));
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
        
        const resolution = imgElement.closest('.gallery-item').querySelector('.resolution');
        resolutionText.textContent = resolution ? resolution.textContent : '';
        
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
    
    // 事件监听
    document.querySelectorAll('.gallery-image, .before-image, .after-image').forEach(img => {
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
});