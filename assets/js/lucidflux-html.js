// LucidFlux HTML Script - Modern Interactive Features
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
    
    // Scroll animations and reveal effects
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, observerOptions);
    
    // Observe all sections and cards
    document.querySelectorAll('section, .feature-card, .framework-card, .metric-card, .application-card, .gallery-item').forEach(el => {
        el.classList.add('scroll-reveal');
        observer.observe(el);
    });
    
    // Floating demo images interaction
    const demoImages = document.querySelectorAll('.demo-image');
    
    // Add scatter effect on scroll
    let scrollTimeout;
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const speed = 0.5;
        
        demoImages.forEach((img, index) => {
            const yOffset = -(scrolled * speed * (index + 1) * 0.3);
            const xOffset = Math.sin(scrolled * 0.01 + index) * 20;
            const rotation = Math.sin(scrolled * 0.005 + index) * 5;
            
            img.style.transform = `translateY(${yOffset}px) translateX(${xOffset}px) rotate(${rotation}deg)`;
        });
        
        // Clear timeout to prevent too many updates
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
            demoImages.forEach(img => {
                img.style.transition = 'transform 0.3s ease';
            });
        }, 100);
    });
    
    // Video player functionality
    const videoPlaceholder = document.querySelector('.video-placeholder');
    const playButton = document.querySelector('.play-button');
    const videoControls = document.querySelector('.video-controls');
    const progressBar = document.querySelector('.progress');
    const playPauseBtn = document.querySelector('.play-pause');
    const volumeBtn = document.querySelector('.volume');
    const fullscreenBtn = document.querySelector('.fullscreen');
    
    let isPlaying = false;
    let currentProgress = 35;
    
    if (videoPlaceholder && playButton) {
        videoPlaceholder.addEventListener('click', function() {
            if (!isPlaying) {
                // Simulate video playing
                isPlaying = true;
                playButton.innerHTML = '<i class="fas fa-pause"></i>';
                startVideoProgress();
            } else {
                // Simulate video pausing
                isPlaying = false;
                playButton.innerHTML = '<i class="fas fa-play"></i>';
                stopVideoProgress();
            }
        });
    }
    
    function startVideoProgress() {
        const progressInterval = setInterval(() => {
            if (!isPlaying || currentProgress >= 100) {
                clearInterval(progressInterval);
                if (currentProgress >= 100) {
                    isPlaying = false;
                    playButton.innerHTML = '<i class="fas fa-play"></i>';
                    currentProgress = 0;
                    progressBar.style.width = '0%';
                }
                return;
            }
            currentProgress += 0.5;
            progressBar.style.width = currentProgress + '%';
        }, 100);
    }
    
    function stopVideoProgress() {
        // Progress stops when paused
    }
    
    // Control buttons functionality
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', function() {
            videoPlaceholder.click();
        });
    }
    
    if (volumeBtn) {
        volumeBtn.addEventListener('click', function() {
            const icon = this.querySelector('i');
            if (icon.classList.contains('fa-volume-up')) {
                icon.className = 'fas fa-volume-mute';
            } else {
                icon.className = 'fas fa-volume-up';
            }
        });
    }
    
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', function() {
            const videoPlayer = document.querySelector('.video-player');
            if (videoPlayer) {
                if (videoPlayer.requestFullscreen) {
                    videoPlayer.requestFullscreen();
                } else if (videoPlayer.webkitRequestFullscreen) {
                    videoPlayer.webkitRequestFullscreen();
                } else if (videoPlayer.msRequestFullscreen) {
                    videoPlayer.msRequestFullscreen();
                }
            }
        });
    }
    
    // Gallery Modal functionality
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    const closeBtn = document.querySelector('.close');
    const galleryItems = document.querySelectorAll('.gallery-item');
    const prevBtn = document.querySelector('.modal-nav.prev');
    const nextBtn = document.querySelector('.modal-nav.next');
    
    let currentImageIndex = 0;
    
    // Open modal when clicking on gallery items
    galleryItems.forEach((item, index) => {
        item.addEventListener('click', function() {
            const img = this.querySelector('img');
            const overlayContent = this.querySelector('.overlay-content');
            
            modal.style.display = 'block';
            modalImg.src = img.src;
            modalImg.alt = img.alt;
            modalCaption.textContent = overlayContent.querySelector('h4').textContent;
            currentImageIndex = index;
            
            // Disable body scroll
            document.body.style.overflow = 'hidden';
            
            // Add entrance animation
            setTimeout(() => {
                modalImg.style.opacity = '1';
            }, 50);
        });
    });
    
    // Close modal
    function closeModal() {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        modalImg.style.opacity = '0';
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', closeModal);
    }
    
    // Close modal when clicking outside the image
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });
    
    // Navigation buttons
    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            currentImageIndex = (currentImageIndex - 1 + galleryItems.length) % galleryItems.length;
            showModalImage(currentImageIndex);
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            currentImageIndex = (currentImageIndex + 1) % galleryItems.length;
            showModalImage(currentImageIndex);
        });
    }
    
    // Keyboard navigation for gallery
    document.addEventListener('keydown', function(e) {
        if (modal.style.display === 'block') {
            if (e.key === 'ArrowLeft') {
                currentImageIndex = (currentImageIndex - 1 + galleryItems.length) % galleryItems.length;
                showModalImage(currentImageIndex);
            } else if (e.key === 'ArrowRight') {
                currentImageIndex = (currentImageIndex + 1) % galleryItems.length;
                showModalImage(currentImageIndex);
            }
        }
    });
    
    function showModalImage(index) {
        const item = galleryItems[index];
        const img = item.querySelector('img');
        const overlayContent = item.querySelector('.overlay-content');
        
        // Fade out effect
        modalImg.style.opacity = '0';
        
        setTimeout(() => {
            modalImg.src = img.src;
            modalImg.alt = img.alt;
            modalCaption.textContent = overlayContent.querySelector('h4').textContent;
            modalImg.style.opacity = '1';
        }, 200);
    }
    
    // Like button functionality
    const likeButtons = document.querySelectorAll('.like-btn');
    likeButtons.forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const icon = this.querySelector('i');
            if (icon.classList.contains('far')) {
                icon.className = 'fas fa-heart';
                this.style.color = '#e74c3c';
            } else {
                icon.className = 'far fa-heart';
                this.style.color = '';
            }
        });
    });
    
    // Parallax effect for hero section
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const heroSection = document.querySelector('.hero-section');
        const orbs = document.querySelectorAll('.orb');
        
        if (heroSection) {
            const yPos = -(scrolled * 0.5);
            heroSection.style.transform = `translateY(${yPos}px)`;
        }
        
        // Parallax for floating orbs
        orbs.forEach((orb, index) => {
            const speed = 0.3 + (index * 0.1);
            const yPos = -(scrolled * speed);
            orb.style.transform += ` translateY(${yPos}px)`;
        });
    });
    
    // Dynamic orb interaction
    const orbs = document.querySelectorAll('.orb');
    orbs.forEach(orb => {
        orb.addEventListener('mouseenter', function() {
            this.style.transform += ' scale(1.2)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        orb.addEventListener('mouseleave', function() {
            this.style.transform = this.style.transform.replace(' scale(1.2)', '');
        });
    });
    
    // Performance metrics animation
    const metricCards = document.querySelectorAll('.metric-card');
    const metricsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target.querySelector('.progress-bar');
                if (progressBar) {
                    const width = progressBar.style.width;
                    progressBar.style.width = '0%';
                    setTimeout(() => {
                        progressBar.style.width = width;
                    }, 200);
                }
            }
        });
    }, { threshold: 0.5 });
    
    metricCards.forEach(card => {
        metricsObserver.observe(card);
    });
    
    // Touch support for mobile devices
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', function(e) {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    document.addEventListener('touchend', function(e) {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
    
    function handleSwipe() {
        if (modal.style.display === 'block') {
            const swipeThreshold = 50;
            const diff = touchStartX - touchEndX;
            
            if (Math.abs(diff) > swipeThreshold) {
                if (diff > 0) {
                    // Swipe left - next image
                    currentImageIndex = (currentImageIndex + 1) % galleryItems.length;
                    showModalImage(currentImageIndex);
                } else {
                    // Swipe right - previous image
                    currentImageIndex = (currentImageIndex - 1 + galleryItems.length) % galleryItems.length;
                    showModalImage(currentImageIndex);
                }
            }
        }
    }
    
    // Performance optimization: throttle scroll events
    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }
    
    // Apply throttling to scroll events
    const throttledScrollHandler = throttle(function() {
        // Handle scroll-based animations
        const scrolled = window.pageYOffset;
        
        // Update floating elements
        demoImages.forEach((img, index) => {
            const yOffset = -(scrolled * 0.5 * (index + 1) * 0.3);
            const xOffset = Math.sin(scrolled * 0.01 + index) * 20;
            img.style.transform = `translateY(${yOffset}px) translateX(${xOffset}px)`;
        });
    }, 16); // ~60fps
    
    window.addEventListener('scroll', throttledScrollHandler);
    
    // Add print styles
    window.addEventListener('beforeprint', function() {
        document.body.classList.add('print-mode');
    });
    
    window.addEventListener('afterprint', function() {
        document.body.classList.remove('print-mode');
    });
    
    // Console welcome message
    console.log('%cLucidFlux 🌊', 'color: #667eea; font-size: 24px; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);');
    console.log('%cAdvanced Image Generation with Flow Matching', 'color: #764ba2; font-size: 16px; font-weight: 500;');
    console.log('%cModern Web Experience with Glassmorphism Design', 'color: #f093fb; font-size: 12px; font-style: italic;');
    console.log('%cGitHub: https://github.com/Ephemeral182/PosterCraft', 'color: #27ae60; font-size: 11px;');
    
    // Initialize page animations
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 100);
    
});