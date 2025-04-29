(function() {
    // --- Configuration ---
    const chatIconColor = '#1a1a1a'; // Example: Dark color for icon background
    const chatIconContent = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" width="28px" height="28px">
          <path d="M0 0h24v24H0z" fill="none"/>
          <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
        </svg>`; // SVG Icon (can be text or emoji too like '💬')

    const panelHeaderTitle = ''; // Removed title text
    const panelBackgroundColor = '#f8f9fa'; // Light background for panel
    const panelHeaderColor = '#e9ecef'; // Slightly different header bg
    const panelWidth = '400px'; // Increased width
    const panelHeight = '500px'; // Adjust as needed
    const sendButtonText = 'Send'; // Changed button text
    const inputPlaceholder = 'Type your message...';
    const sendButtonIcon = `
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20px" height="20px">
          <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z"/>
        </svg>`; // Send icon SVG

    // --- Updated Popular Questions for Gamer Product Websites ---
    const popularQuestions = [
        { text: 'What monitor is best for competitive FPS?', url: '#' },
        { text: 'How do I choose the right gaming chair?', url: '#' },
        { text: 'Mechanical vs Membrane keyboards for gaming?', url: '#' },
        { text: 'Where can I find PS5 stock updates?', url: '#' },
        { text: "What's the return policy on peripherals?", url: '#' }
    ];

    // --- Sample Autocomplete Data ---
    const autocompleteSuggestions = [
        "How do I reset my password?",
        "What are your pricing plans?",
        "How can I contact support?",
        "Where is my invoice?",
        "How to upgrade my account?",
        "Can I get a demo?",
        "What features do you offer?"
    ];

    // --- CSS Styles ---
    const styles = `
        :root {
            --chat-widget-panel-width: ${panelWidth};
            --chat-widget-panel-height: ${panelHeight};
            --chat-widget-icon-color: ${chatIconColor};
            --chat-widget-panel-bg: ${panelBackgroundColor};
            --chat-widget-panel-header-bg: ${panelHeaderColor};
        }

        #chat-widget-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }

        #chat-widget-icon {
            background-color: var(--chat-widget-icon-color);
            color: white;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            display: flex;
            justify-content: center;
            align-items: center;
            transition: transform 0.2s ease;
        }

        #chat-widget-icon:hover {
            transform: scale(1.1);
        }

        #chat-widget-panel {
            position: absolute;
            bottom: 70px; /* Position above the icon */
            right: 0;
            width: var(--chat-widget-panel-width);
            max-height: var(--chat-widget-panel-height);
            height: var(--chat-widget-panel-height); /* Fixed height */
            background-color: var(--chat-widget-panel-bg);
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            overflow: hidden;
            display: none; /* Hidden by default */
            flex-direction: column;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            font-size: 15px; /* Base font size for rem units */
        }

        #chat-widget-panel.open {
            display: flex;
        }

        .chat-widget-header {
            background-color: var(--chat-widget-panel-header-bg);
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0; /* Prevent header shrinking */
        }

        /* Style for the logo in the header */
        .chat-widget-header img {
            height: 24px; /* Adjust size as needed */
            margin-right: 10px; /* Space between logo and title */
        }

        .chat-widget-header h3 {
            margin: 0;
            font-size: 1.1em;
            color: #333;
        }

         .chat-widget-close-btn {
             background: none;
             border: none;
             font-size: 1.5em;
             cursor: pointer;
             color: #6c757d;
             padding: 0 5px;
         }
         .chat-widget-close-btn:hover {
             color: #333;
         }

        .chat-widget-body {
            flex-grow: 1; /* Takes available vertical space */
            overflow-y: auto; /* Scroll if content overflows */
            padding: 15px 0px; /* Reset padding, children will handle it */
            background-color: #fff; /* White background for message area */
            display: flex;
            flex-direction: column;
        }

        /* Updated styles for popular questions list */
         .chat-widget-body h4 {
             font-size: 0.75rem; /* Smaller size (e.g., 11-12px) */
             color: #6c757d;
             margin: 0 0 10px 0; /* Adjusted margin */
             text-transform: uppercase;
             font-weight: normal;
             padding: 0 20px; /* Add padding to match body */
             letter-spacing: 0.5px; /* Slight letter spacing */
         }

        .chat-widget-questions {
            list-style: none;
            padding: 0;
            margin: 0 0 15px 0; /* Margin below questions */
            /* border-top: 1px solid #eee; */ /* Removed top border */
            /* border-bottom: 1px solid #eee; */ /* Removed bottom border */
        }

        .chat-widget-questions li a {
            display: flex; /* Use flex for icon alignment */
            align-items: center;
            /* justify-content: space-between; */ /* Removed for simpler layout */
            padding: 10px 20px; /* Slightly reduced padding */
            color: #333;
            text-decoration: none;
            font-size: 0.95rem; /* Slightly smaller than base (e.g., ~14px) */
            border-bottom: 1px solid #eee; /* Add separator line below each */
            transition: background-color 0.2s ease;
            gap: 8px; /* Space between icon and text */
        }
         .chat-widget-questions li:last-child a {
              border-bottom: none; /* No line after the last item */
          }

        .chat-widget-questions li a:hover {
            background-color: #e9ecef;
        }

        /* Style for the question icon (optional) */
        .chat-widget-question-icon {
            width: 16px; /* Adjust size */
            height: 16px;
            fill: #6c757d; /* Icon color */
            flex-shrink: 0; /* Prevent icon shrinking */
        }

        #chat-widget-messages {
            flex-grow: 1; /* Fill available space */
            /* Styles for individual messages would go here */
            padding: 0 20px; /* Padding for messages */
            margin-bottom: 10px; /* Space above footer */
            line-height: 1.5; /* Improve message readability */
        }

        /* Default style for messages added to the chat */
        #chat-widget-messages p {
            font-size: 1rem; /* Base font size (e.g., 15px) */
            margin: 0 0 0.8em 0; /* Spacing between messages */
            word-wrap: break-word; /* Prevent long words overflowing */
        }

        /* Suggestions container positioning relative to footer */
        #chat-widget-suggestions {
            position: absolute;
            bottom: 100%; /* Position right above the footer */
            left: 15px; /* Align with footer padding */
            right: 15px; /* Align with footer padding */
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-bottom: none; /* Connects visually with footer border */
            border-radius: 8px 8px 0 0; /* Rounded top corners */
            box-shadow: 0 -4px 10px rgba(0,0,0,0.1);
            max-height: 180px; /* Limit suggestion height */
            overflow-y: auto;
            z-index: 1001; /* Ensure suggestions are above other panel content */
            display: none; /* Hidden by default */
            margin-bottom: 1px; /* Tiny gap above footer */
        }

        #chat-widget-suggestions ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        #chat-widget-suggestions li {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
            font-size: 0.9rem; /* Smaller than base (e.g., ~13.5px) */
            color: #333;
        }
        #chat-widget-suggestions li:last-child {
            border-bottom: none;
        }

        #chat-widget-suggestions li:hover {
            background-color: #f8f9fa;
        }

        .chat-widget-footer {
            position: relative; /* Needed for absolute positioning of suggestions */
            padding: 10px 15px;
            border-top: 1px solid #dee2e6;
            background-color: var(--chat-widget-panel-bg); /* Match panel bg */
            display: flex; /* Use flexbox for input/button */
            align-items: flex-end; /* Align items to bottom for multi-line text */
            flex-shrink: 0; /* Prevent footer shrinking */
            gap: 8px; /* Gap between input and button */
        }

        #chat-widget-input {
            flex-grow: 1; /* Input takes most space */
            border: 1px solid #ced4da;
            border-radius: 18px; /* More rounded corners */
            padding: 10px 18px; /* Increased padding */
            font-size: 1rem; /* Base font size (e.g., 15px) */
            resize: none; /* Prevent manual resize */
            min-height: 48px; /* Increased minimum height */
            max-height: 150px; /* Increased max height before scrolling */
            height: 48px; /* Initial height, adjusted via JS */
            box-sizing: border-box; /* Include padding/border in height */
            line-height: 1.4; /* Better line spacing */
            overflow-y: auto; /* Allow scrolling if max-height is reached */
        }
         #chat-widget-input:focus {
             outline: none;
             border-color: #86b7fe; /* Highlight on focus */
             box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, .25);
         }

        #chat-widget-send-btn {
            background-color: #007bff; /* Changed back to blue */
            color: white; /* Icon color */
            border: none;
            border-radius: 50%; /* Make it circular */
            width: 36px; /* Fixed width */
            height: 36px; /* Fixed height */
            padding: 0; /* Remove padding */
            font-size: 0.95em;
            cursor: pointer;
            transition: background-color 0.2s ease;
            flex-shrink: 0; /* Prevent button shrinking */
            display: flex; /* Center icon */
            justify-content: center; /* Center icon */
            align-items: center; /* Center icon */
        }

        #chat-widget-send-btn:hover {
            background-color: #0056b3; /* Darker blue shade on hover */
        }
         #chat-widget-send-btn:disabled {
             background-color: #e0e0e0; /* Grey out when disabled */
             cursor: not-allowed;
         }
         #chat-widget-send-btn svg {
             display: block; /* Ensure SVG behaves correctly */
         }
    `;

    // --- Dynamic Element Creation ---
    function createWidget() {
        // 1. Inject CSS
        const styleSheet = document.getElementById('chat-widget-styles');
        if (styleSheet) {
            styleSheet.textContent = styles;
        } else {
            const styleEl = document.createElement('style');
            styleEl.textContent = styles;
            document.head.appendChild(styleEl);
        }

        // 2. Create main container
        const container = document.createElement('div');
        container.id = 'chat-widget-container';

        // 3. Create chat icon
        const iconButton = document.createElement('button');
        iconButton.id = 'chat-widget-icon';
        iconButton.innerHTML = chatIconContent; // Use SVG or text
        iconButton.setAttribute('aria-label', 'Open Chat');

        // 4. Create chat panel
        const panel = document.createElement('div');
        panel.id = 'chat-widget-panel';

        // 5. Create panel header
        const header = document.createElement('div');
        header.className = 'chat-widget-header';
        const headerTitle = document.createElement('h3');
        headerTitle.textContent = panelHeaderTitle;

        // Create and add the logo image
        const logoImg = document.createElement('img');
        logoImg.src = 'https://ad.net/wp-content/themes/wp-adnet/assets/images/adnet.svg';
        logoImg.alt = 'Logo'; // Add alt text for accessibility

        // Container for logo and title to group them
        const logoTitleContainer = document.createElement('div');
        logoTitleContainer.style.display = 'flex';
        logoTitleContainer.style.alignItems = 'center';
        logoTitleContainer.appendChild(logoImg);
        logoTitleContainer.appendChild(headerTitle);

        const closeButton = document.createElement('button');
        closeButton.className = 'chat-widget-close-btn';
        closeButton.innerHTML = '×'; // Simple 'x' close button
        closeButton.setAttribute('aria-label', 'Close Chat');

        // Add the logo/title group and the close button to the header
        header.appendChild(logoTitleContainer);
        header.appendChild(closeButton);

        // 6. Create panel body
        const body = document.createElement('div');
        body.className = 'chat-widget-body';

        // --- Re-add Popular Questions --- 
        const questionsTitle = document.createElement('h4');
        questionsTitle.textContent = 'Popular Questions'; // Or another suitable title
        body.appendChild(questionsTitle);

        const questionList = document.createElement('ul');
        questionList.className = 'chat-widget-questions';

        // Add question icon SVG (optional)
        const questionIconSvg = `
            <svg class="chat-widget-question-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
                <path fill-rule="evenodd" d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                <path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z"/>
            </svg>`;

        popularQuestions.forEach(q => {
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = q.url;
            // Set text and optionally handle clicks differently for chat
            link.textContent = q.text;
            link.target = '_blank'; // Keep original behavior or change
            
            // Prepend the icon SVG to the link
            link.innerHTML = questionIconSvg + link.innerHTML;

            listItem.appendChild(link);
            questionList.appendChild(listItem);
        });
        body.appendChild(questionList);
        // --- End Popular Questions --- 

        const messagesArea = document.createElement('div');
        messagesArea.id = 'chat-widget-messages';
        body.appendChild(messagesArea);

        // 7. Create panel footer (for input and send button)
        const footer = document.createElement('div');
        footer.className = 'chat-widget-footer';

        // 7.5 Create Suggestions container (appended to footer)
        const suggestionsContainer = document.createElement('div');
        suggestionsContainer.id = 'chat-widget-suggestions';
        const suggestionsList = document.createElement('ul');
        suggestionsContainer.appendChild(suggestionsList);
        footer.appendChild(suggestionsContainer); // Append to footer

        const chatInput = document.createElement('textarea');
        chatInput.id = 'chat-widget-input';
        chatInput.placeholder = inputPlaceholder;
        chatInput.rows = 1; // Start with one row, can expand with CSS or JS later
        const sendButton = document.createElement('button');
        sendButton.id = 'chat-widget-send-btn'; // New ID for clarity
        sendButton.innerHTML = sendButtonIcon; // Use SVG Icon
        sendButton.setAttribute('aria-label', 'Send Message'); // Accessibility
        sendButton.disabled = true; // Disabled by default

        footer.appendChild(chatInput);
        footer.appendChild(sendButton);

        // 8. Assemble panel
        panel.appendChild(header);
        panel.appendChild(body);
        panel.appendChild(footer);

        // 9. Assemble container
        container.appendChild(iconButton);
        container.appendChild(panel);

        // 10. Append to body
        document.body.appendChild(container);

        // --- Event Listeners ---
        function togglePanel() {
            panel.classList.toggle('open');
        }

        iconButton.addEventListener('click', togglePanel);
        closeButton.addEventListener('click', togglePanel); // Close button hides panel

        // Action for the new send button
        sendButton.addEventListener('click', () => {
            const messageText = chatInput.value.trim();
            if (messageText) {
                console.log('Sending message:', messageText);
                // Add message to the messagesArea (basic example)
                const messageElement = document.createElement('p');
                messageElement.textContent = `You: ${messageText}`; // Simple display
                 messageElement.style.textAlign = 'right'; // Align user message
                messagesArea.appendChild(messageElement);
                messagesArea.scrollTop = messagesArea.scrollHeight; // Scroll to bottom
                chatInput.value = ''; // Clear input
                sendButton.disabled = true; // Disable button after sending
                chatInput.style.height = '48px'; // Reset height after sending
                suggestionsContainer.style.display = 'none'; // Hide suggestions
                // Here you would typically send the message to a backend
            }
        });

        // Enable/disable send button AND handle autocomplete based on input
        chatInput.addEventListener('input', () => {
            sendButton.disabled = chatInput.value.trim().length === 0;

            // Auto-resize textarea height
            chatInput.style.height = 'auto'; // Temporarily shrink
            const scrollHeight = chatInput.scrollHeight;
            const maxHeight = parseInt(window.getComputedStyle(chatInput).maxHeight, 10);
            const minHeight = parseInt(window.getComputedStyle(chatInput).minHeight, 10);

            let newHeight = Math.max(minHeight, scrollHeight);
            if (newHeight > maxHeight) {
                newHeight = maxHeight;
             }
            chatInput.style.height = `${newHeight}px`;

            // --- Autocomplete Logic ---
            const currentInput = chatInput.value.toLowerCase().trim();
            suggestionsList.innerHTML = ''; // Clear previous suggestions

            if (currentInput.length > 0) {
                const filteredSuggestions = autocompleteSuggestions.filter(s =>
                    s.toLowerCase().includes(currentInput)
                );

                if (filteredSuggestions.length > 0) {
                    filteredSuggestions.forEach(suggestionText => {
                        const li = document.createElement('li');
                        li.textContent = suggestionText;
                        li.addEventListener('click', () => {
                            chatInput.value = suggestionText; // Fill input
                            suggestionsContainer.style.display = 'none'; // Hide suggestions
                            sendButton.disabled = false; // Enable send button
                            chatInput.focus(); // Keep focus on input
                            // Trigger input event again to resize textarea properly
                            chatInput.dispatchEvent(new Event('input'));
                        });
                        suggestionsList.appendChild(li);
                    });
                    suggestionsContainer.style.display = 'block'; // Show suggestions
                } else {
                    suggestionsContainer.style.display = 'none'; // Hide if no matches
                }
            } else {
                suggestionsContainer.style.display = 'none'; // Hide if input is empty
            }
        });


        // Optional: Send message on Enter key press in textarea
        chatInput.addEventListener('keypress', function(event) {
            // Check if Enter is pressed (without Shift for newline)
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault(); // Prevent default newline insertion
                sendButton.click(); // Trigger send button click
            }
        });

        // Hide suggestions when clicking outside the input/suggestions area
        document.addEventListener('click', function(event) {
            if (!container.contains(event.target) && panel.classList.contains('open')) {
                 // Uncomment below line to close panel when clicking outside
                 // togglePanel();
            }
            // Hide suggestions if click is outside the footer AND outside the suggestions box itself
            if (!footer.contains(event.target) && !suggestionsContainer.contains(event.target)) {
                suggestionsContainer.style.display = 'none';
            }
        });

    }

    // --- Initialization ---
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createWidget);
    } else {
        // DOMContentLoaded has already fired
        createWidget();
    }

})(); 