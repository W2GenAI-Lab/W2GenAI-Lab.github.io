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
        document.querySelectorAll('.style-gallery').forEach(gallery => {
            const mainImage = gallery.querySelector('.style-main .style-image');
            const images = Array.from(gallery.querySelectorAll('.thumb'));
            const prevBtn = gallery.querySelector('.style-nav.prev');
            const nextBtn = gallery.querySelector('.style-nav.next');
            
            if (images.length === 0) return;
            
            let currentIndex = 0;
            
            function showImage(index) {
                // 更新主图
                const newImageSrc = images[index].src;
                const newImageAlt = images[index].alt;
                
                // 创建新的主图元素
                const newMainImage = mainImage.cloneNode(true);
                newMainImage.src = newImageSrc;
                newMainImage.alt = newImageAlt;
                
                // 替换主图
                mainImage.parentNode.replaceChild(newMainImage, mainImage);
                
                // 更新缩略图状态
                images.forEach(img => img.classList.remove('active'));
                images[index].classList.add('active');
                
                currentIndex = index;
                
                // 添加点击事件到新的主图
                newMainImage.addEventListener('click', () => {
                    currentImageIndex = galleryImages.indexOf(newMainImage);
                    showModal(newMainImage);
                });
            }
            
            prevBtn.addEventListener('click', () => {
                const newIndex = (currentIndex - 1 + images.length) % images.length;
                showImage(newIndex);
            });
            
            nextBtn.addEventListener('click', () => {
                const newIndex = (currentIndex + 1) % images.length;
                showImage(newIndex);
            });
            
            // 缩略图点击事件
            images.forEach((thumb, index) => {
                thumb.addEventListener('click', () => {
                    showImage(index);
                });
            });
            
            // 主图点击事件（模态框）
            mainImage.addEventListener('click', () => {
                currentImageIndex = galleryImages.indexOf(mainImage);
                showModal(mainImage);
            });
            
            // 键盘导航
            gallery.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') prevBtn.click();
                if (e.key === 'ArrowRight') nextBtn.click();
            });
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