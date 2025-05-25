// Language switching functionality
document.addEventListener('DOMContentLoaded', function() {
    // Set initial language based on user preference or default to English
    const userLang = navigator.language || navigator.userLanguage;
    const initialLang = userLang.startsWith('ar') ? 'ar' : 'en';
    setLanguage(initialLang);

    // Add click handlers for language switchers
    document.querySelectorAll('.lang-switch').forEach(button => {
        button.addEventListener('click', function() {
            const lang = this.getAttribute('data-lang');
            setLanguage(lang);
        });
    });
});

function setLanguage(lang) {
    // Hide all content containers
    document.querySelectorAll('.content-container').forEach(container => {
        container.classList.remove('active');
    });

    // Show content for selected language
    document.querySelector(`.content-container.${lang}`).classList.add('active');

    // Update language switcher buttons
    document.querySelectorAll('.lang-switch').forEach(button => {
        button.classList.remove('active');
        if (button.getAttribute('data-lang') === lang) {
            button.classList.add('active');
        }
    });

    // Update document direction
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
    document.documentElement.lang = lang;

    // Update loading text
    document.querySelectorAll('.loading-text').forEach(text => {
        text.style.display = 'none';
    });
    document.querySelector(`.loading-text.${lang}`).style.display = 'block';

    // Update copyright text
    document.querySelectorAll('.copyright p').forEach(text => {
        text.style.display = 'none';
    });
    document.querySelector(`.copyright p.${lang}`).style.display = 'block';
} 