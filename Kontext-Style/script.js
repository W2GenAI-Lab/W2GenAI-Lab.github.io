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
    
    // Style Gallery 滚动播放功能
    initStyleScrolling();
});

// Style Gallery 滚动播放功能
function initStyleScrolling() {
    const styleContainers = document.querySelectorAll('.style-scroll-container');
    
    styleContainers.forEach(container => {
        const scroll = container.querySelector('.style-scroll');
        const images = scroll.querySelectorAll('.scroll-image');
        
        if (images.length === 0) return;
        
        // 克隆图片以实现无缝滚动
        const clones = Array.from(images).map(img => img.cloneNode(true));
        clones.forEach(clone => scroll.appendChild(clone));
        
        // 添加导航按钮
        const prevBtn = document.createElement('button');
        prevBtn.className = 'scroll-nav prev';
        prevBtn.innerHTML = '‹';
        prevBtn.addEventListener('click', () => scrollImages(scroll, -1));
        
        const nextBtn = document.createElement('button');
        nextBtn.className = 'scroll-nav next';
        nextBtn.innerHTML = '›';
        nextBtn.addEventListener('click', () => scrollImages(scroll, 1));
        
        container.appendChild(prevBtn);
        container.appendChild(nextBtn);
        
        // 添加指示点
        const dotsContainer = document.createElement('div');
        dotsContainer.className = 'scroll-dots';
        
        for (let i = 0; i < images.length; i++) {
            const dot = document.createElement('span');
            dot.className = 'dot';
            if (i === 0) dot.classList.add('active');
            dot.addEventListener('click', () => scrollToImage(scroll, i));
            dotsContainer.appendChild(dot);
        }
        
        container.appendChild(dotsContainer);
        
        // 自动滚动
        let autoScrollInterval = setInterval(() => {
            scrollImages(scroll, 1);
        }, 3000);
        
        // 鼠标悬停时暂停自动滚动
        container.addEventListener('mouseenter', () => {
            clearInterval(autoScrollInterval);
        });
        
        container.addEventListener('mouseleave', () => {
            autoScrollInterval = setInterval(() => {
                scrollImages(scroll, 1);
            }, 3000);
        });
        
        // 图片点击放大
        images.forEach(img => {
            img.addEventListener('click', function() {
                showImageModal(this.src, this.alt);
            });
        });
    });
}

// 滚动到指定图片
function scrollToImage(scroll, index) {
    const imageWidth = scroll.querySelector('.scroll-image').offsetWidth + 10; // 包含gap
    const targetPosition = -index * imageWidth;
    scroll.style.transform = `translateX(${targetPosition}px)`;
    
    // 更新指示点
    const container = scroll.closest('.style-scroll-container');
    const dots = container.querySelectorAll('.dot');
    dots.forEach((dot, i) => {
        dot.classList.toggle('active', i === index);
    });
}

// 滚动图片
function scrollImages(scroll, direction) {
    const images = scroll.querySelectorAll('.scroll-image');
    const originalCount = images.length / 2; // 因为有克隆
    const imageWidth = scroll.querySelector('.scroll-image').offsetWidth + 10;
    
    const currentTransform = scroll.style.transform;
    const currentX = currentTransform ? parseInt(currentTransform.match(/-?\d+/)[0]) : 0;
    
    let newX = currentX - (direction * imageWidth);
    
    // 边界检查和循环
    if (newX < -originalCount * imageWidth) {
        newX = 0;
    } else if (newX > 0) {
        newX = -originalCount * imageWidth + imageWidth;
    }
    
    scroll.style.transform = `translateX(${newX}px)`;
    
    // 更新指示点
    const container = scroll.closest('.style-scroll-container');
    const dots = container.querySelectorAll('.dot');
    const currentIndex = Math.abs(Math.round(newX / imageWidth)) % originalCount;
    dots.forEach((dot, i) => {
        dot.classList.toggle('active', i === currentIndex);
    });
}

// 显示图片模态框
function showImageModal(src, alt) {
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');
    const captionText = document.getElementById('caption');
    
    modal.style.display = 'block';
    modalImg.style.backgroundImage = `url(${src})`;
    captionText.textContent = alt;
    
    // 清除分辨率文本，因为滚动图片没有分辨率信息
    const resolutionText = document.getElementById('resolution');
    resolutionText.textContent = '';
}