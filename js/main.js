document.addEventListener('DOMContentLoaded', () => {
    // Loading Screen
    const loading = document.querySelector('.loading');
    window.addEventListener('load', () => {
        setTimeout(() => {
            loading.style.opacity = '0';
            setTimeout(() => {
                loading.style.display = 'none';
            }, 500);
        }, 1500);
    });

    // Theme Toggle
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme') || 'light';
    body.classList.add(`${savedTheme}-mode`);
    
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        body.classList.toggle('light-mode');
        localStorage.setItem('theme', body.classList.contains('dark-mode') ? 'dark' : 'light');
    });
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe sections and service cards
    document.querySelectorAll('section, .service-card').forEach(element => {
        observer.observe(element);
    });

    // Form submission handling
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(contactForm);
            const data = Object.fromEntries(formData);
            
            // Here you would typically send the data to a server
            console.log('Form submitted:', data);
            
            // Show success message
            const submitButton = contactForm.querySelector('.submit-button');
            const originalText = submitButton.textContent;
            submitButton.textContent = 'Message Sent!';
            submitButton.style.backgroundColor = '#4CAF50';
            
            // Reset form
            contactForm.reset();
            
            // Reset button after 3 seconds
            setTimeout(() => {
                submitButton.textContent = originalText;
                submitButton.style.backgroundColor = '';
            }, 3000);
        });
    }

    // Parallax effect for hero section
    const hero = document.querySelector('.hero');
    if (hero) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            hero.style.backgroundPositionY = `${scrolled * 0.5}px`;
        });
    }

    // Mobile menu toggle (for smaller screens)
    const createMobileMenu = () => {
        const nav = document.querySelector('nav');
        const navLinks = document.querySelector('.nav-links');
        const menuButton = document.createElement('button');
        menuButton.className = 'mobile-menu-button';
        menuButton.innerHTML = '<i class="fas fa-bars"></i>';
        menuButton.setAttribute('aria-label', 'Toggle menu');
        
        nav.insertBefore(menuButton, navLinks);
        
        menuButton.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuButton.innerHTML = navLinks.classList.contains('active') 
                ? '<i class="fas fa-times"></i>' 
                : '<i class="fas fa-bars"></i>';
        });
    };

    // Initialize mobile menu if screen is small
    if (window.innerWidth <= 768) {
        createMobileMenu();
    }

    // Handle window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            if (!document.querySelector('.mobile-menu-button')) {
                createMobileMenu();
            }
        } else {
            const menuButton = document.querySelector('.mobile-menu-button');
            if (menuButton) {
                menuButton.remove();
                document.querySelector('.nav-links').classList.remove('active');
            }
        }
    });

    // Add hover effect to service cards
    const serviceCards = document.querySelectorAll('.service-card');
    serviceCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-10px)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });

    // Add scroll progress indicator
    const createScrollProgress = () => {
        const progressBar = document.createElement('div');
        progressBar.className = 'scroll-progress';
        document.body.appendChild(progressBar);
        
        window.addEventListener('scroll', () => {
            const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (winScroll / height) * 100;
            progressBar.style.width = scrolled + '%';
        });
    };

    createScrollProgress();

    // Language Management
    /*
    function setLanguage(lang) {
        // Update HTML direction and language
        html.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
        html.setAttribute('lang', lang);
        localStorage.setItem('language', lang);

        // Toggle content containers
        document.querySelectorAll('.content-container').forEach(container => {
            if (container.classList.contains(lang)) {
                container.style.display = 'block';
                setTimeout(() => container.style.opacity = '1', 50);
            } else {
                container.style.opacity = '0';
                setTimeout(() => container.style.display = 'none', 500);
            }
        });

        // Toggle all language-specific elements
        document.querySelectorAll('.en, .ar').forEach(el => {
            if (el.classList.contains(lang)) {
                el.style.display = el.tagName === 'A' ? 'inline-block' : 'block';
            } else {
                el.style.display = 'none';
            }
        });

        // Update loading text
        document.querySelectorAll('.loading-text').forEach(text => {
            text.style.display = text.classList.contains(lang) ? 'block' : 'none';
        });

        // Update navigation links
        document.querySelectorAll('.nav-links a').forEach(link => {
            if (link.classList.contains(lang)) {
                link.style.display = 'inline-block';
            } else {
                link.style.display = 'none';
            }
        });

        // Update footer groups
        document.querySelectorAll('.link-group, .social-group').forEach(group => {
            if (group.classList.contains(lang)) {
                group.style.display = 'block';
            } else {
                group.style.display = 'none';
            }
        });

        // Update copyright text
        document.querySelectorAll('.copyright p').forEach(text => {
            text.style.display = text.classList.contains(lang) ? 'block' : 'none';
        });
    }
    */
}); 
