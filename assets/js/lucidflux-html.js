// UltraPixel Gallery Script
document.addEventListener('DOMContentLoaded', function() {
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const targetPosition = targetSection.offsetTop - 80;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Modal functionality
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');
    const captionText = document.getElementById('caption');
    const resolutionText = document.getElementById('resolution');
    const thumbnails = document.querySelector('.thumbnails');
    
    let currentImageIndex = 0;
    let galleryImages = [];

    // Collect all gallery images
    function collectGalleryImages() {
        galleryImages = Array.from(document.querySelectorAll('.gallery-image')).map(img => ({
            src: img.src,
            alt: img.alt,
            resolution: img.parentElement.querySelector('.resolution')?.textContent || ''
        }));
    }

    // Open modal when clicking on gallery images
    function openModal(imageSrc, imageAlt, imageResolution) {
        modal.style.display = 'block';
        modalImg.src = imageSrc;
        captionText.textContent = imageAlt;
        resolutionText.textContent = imageResolution;
        
        // Find current image index
        currentImageIndex = galleryImages.findIndex(img => img.src === imageSrc);
        
        // Generate thumbnails
        generateThumbnails();
        
        // Update active thumbnail
        updateActiveThumbnail();
    }

    // Generate thumbnails for modal
    function generateThumbnails() {
        thumbnails.innerHTML = '';
        galleryImages.forEach((image, index) => {
            const thumb = document.createElement('img');
            thumb.src = image.src;
            thumb.alt = image.alt;
            thumb.addEventListener('click', () => {
                currentImageIndex = index;
                modalImg.src = image.src;
                captionText.textContent = image.alt;
                resolutionText.textContent = image.resolution;
                updateActiveThumbnail();
            });
            thumbnails.appendChild(thumb);
        });
    }

    // Update active thumbnail
    function updateActiveThumbnail() {
        const thumbs = thumbnails.querySelectorAll('img');
        thumbs.forEach((thumb, index) => {
            thumb.classList.toggle('active', index === currentImageIndex);
        });
    }

    // Navigate to previous image
    function prevImage() {
        currentImageIndex = (currentImageIndex - 1 + galleryImages.length) % galleryImages.length;
        const image = galleryImages[currentImageIndex];
        modalImg.src = image.src;
        captionText.textContent = image.alt;
        resolutionText.textContent = image.resolution;
        updateActiveThumbnail();
    }

    // Navigate to next image
    function nextImage() {
        currentImageIndex = (currentImageIndex + 1) % galleryImages.length;
        const image = galleryImages[currentImageIndex];
        modalImg.src = image.src;
        captionText.textContent = image.alt;
        resolutionText.textContent = image.resolution;
        updateActiveThumbnail();
    }

    // Close modal
    function closeModal() {
        modal.style.display = 'none';
    }

    // Event listeners
    document.addEventListener('click', function(e) {
        // Gallery image clicks
        if (e.target.classList.contains('gallery-image')) {
            const resolution = e.target.parentElement.querySelector('.resolution')?.textContent || '';
            openModal(e.target.src, e.target.alt, resolution);
        }
        
        // Close button
        if (e.target.classList.contains('close')) {
            closeModal();
        }
        
        // Navigation arrows
        if (e.target.classList.contains('prev')) {
            prevImage();
        }
        
        if (e.target.classList.contains('next')) {
            nextImage();
        }
    });

    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (modal.style.display === 'block') {
            if (e.key === 'Escape') {
                closeModal();
            } else if (e.key === 'ArrowLeft') {
                prevImage();
            } else if (e.key === 'ArrowRight') {
                nextImage();
            }
        }
    });

    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Initialize gallery
    collectGalleryImages();
    
    // Add hover effects to gallery items
    const galleryItems = document.querySelectorAll('.gallery-item');
    galleryItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Lazy loading for images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                observer.unobserve(img);
            }
        });
    });

    images.forEach(img => imageObserver.observe(img));
});