let chatOpen = false;
let isMinimized = false;
let messageHistory = [];
let isApiAvailable = true; // Simulate API availability
let unreadCount = 0;
let userInfo = null;
let chatStarted = false;

// Initialize the chat widget
document.addEventListener('DOMContentLoaded', function() {
    // Check if user info is stored in localStorage
    const storedUserInfo = localStorage.getItem('chatUserInfo');
    if (storedUserInfo) {
        userInfo = JSON.parse(storedUserInfo);
        chatStarted = true;
    }
    
    // Set initial state
    updateChatBadge();
    
    // Add event listeners
    document.getElementById('messageInput').addEventListener('focus', function() {
        this.style.borderColor = 'var(--primary-color)';
    });
    
    document.getElementById('messageInput').addEventListener('blur', function() {
        this.style.borderColor = 'var(--border-color)';
    });
    
    // Simulate receiving a message after 5 seconds
    setTimeout(() => {
        if (!chatOpen) {
            unreadCount++;
            updateChatBadge();
        }
    }, 5000);
});

function toggleChat() {
    const chatWindow = document.getElementById('chatWindow');
    
    chatOpen = !chatOpen;
    
    if (chatOpen) {
        chatWindow.classList.add('active');
        if (isMinimized) {
            chatWindow.classList.add('minimized');
        }
        
        // Show pre-chat form or welcome message based on user info
        if (!userInfo) {
            showPreChatForm();
        } else if (chatStarted) {
            showWelcomeMessage();
            document.getElementById('chatFooter').style.display = 'block';
            setTimeout(() => {
                document.getElementById('messageInput').focus();
            }, 100);
        }
        
        // Reset unread count
        unreadCount = 0;
        updateChatBadge();
    } else {
        chatWindow.classList.remove('active');
    }
}

function showPreChatForm() {
    const messagesContainer = document.getElementById('chatMessages');
    messagesContainer.innerHTML = '';
    
    const formHtml = `
        <div class="pre-chat-form">
            <h3>Before we start</h3>
            <p>Please provide your details so we can assist you better.</p>
            
            <div class="form-group">
                <label for="userName">Full Name *</label>
                <input type="text" id="userName" placeholder="Enter your full name">
                <div class="error-message" id="nameError">Please enter your name</div>
            </div>
            
            <div class="form-group">
                <label for="userEmail">Email Address *</label>
                <input type="email" id="userEmail" placeholder="Enter your email">
                <div class="error-message" id="emailError">Please enter a valid email</div>
            </div>
            
            <div class="form-group">
                <label for="userPhone">Mobile Number *</label>
                <input type="tel" id="userPhone" placeholder="Enter your mobile number">
                <div class="error-message" id="phoneError">Please enter a valid mobile number</div>
            </div>
            
            <button class="form-button" onclick="submitUserInfo()">Start Chat</button>
        </div>
    `;
    
    messagesContainer.innerHTML = formHtml;
    document.getElementById('chatFooter').style.display = 'none';
}

function showWelcomeMessage() {
    const messagesContainer = document.getElementById('chatMessages');
    messagesContainer.innerHTML = '';
    
    // Create welcome message element
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'welcome-message';
    welcomeDiv.innerHTML = `
        <h3>Welcome to our support! 👋</h3>
        <p>How can we help you today? Our team is ready to assist you with any questions or issues you may have.</p>
        <div class="quick-actions">
            <button class="quick-action-btn" onclick="sendQuickMessage('I need help with my order')">Order Help</button>
            <button class="quick-action-btn" onclick="sendQuickMessage('I have a billing question')">Billing</button>
            <button class="quick-action-btn" onclick="sendQuickMessage('I need technical support')">Technical Support</button>
        </div>
    `;
    
    messagesContainer.appendChild(welcomeDiv);
    
    // Add user info message
    if (userInfo) {
        const infoMessage = `Hello, I'm ${userInfo.name}. I need assistance with my account.`;
        addMessage(infoMessage, 'user');
        
        // Show typing indicator and agent response
        showTypingIndicator();
        setTimeout(() => {
            hideTypingIndicator();
            const response = `Hello ${userInfo.name}! Thank you for providing your details. How can I assist you today?`;
            addMessage(response, 'agent');
        }, 2000);
    }
}

async function submitUserInfo() {
    const name = document.getElementById('userName').value.trim();
    const email = document.getElementById('userEmail').value.trim();
    const phone = document.getElementById('userPhone').value.trim();
    
    // Reset error states
    document.getElementById('userName').classList.remove('error');
    document.getElementById('userEmail').classList.remove('error');
    document.getElementById('userPhone').classList.remove('error');
    document.getElementById('nameError').style.display = 'none';
    document.getElementById('emailError').style.display = 'none';
    document.getElementById('phoneError').style.display = 'none';
    
    let isValid = true;
    
    // Validate name
    if (!name) {
        document.getElementById('userName').classList.add('error');
        document.getElementById('nameError').style.display = 'block';
        isValid = false;
    }
    
    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
        document.getElementById('userEmail').classList.add('error');
        document.getElementById('emailError').style.display = 'block';
        isValid = false;
    }
    
    // Validate phone (simple validation for demo)
    const phoneRegex = /^[\d\s\-\+\(\)]+$/;
    if (!phone || !phoneRegex.test(phone) || phone.length < 10) {
        document.getElementById('userPhone').classList.add('error');
        document.getElementById('phoneError').style.display = 'block';
        isValid = false;
    }
    
    if (!isValid) return;
    
    try {
        // Send user info to backend API
        const response = await fetch('/start_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                email: email,
                mobile: phone
            }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to start chat');
        }
        
        // Save user info
        userInfo = {
            name: name,
            email: email,
            phone: phone
        };
        
        // Store in localStorage
        localStorage.setItem('chatUserInfo', JSON.stringify(userInfo));
        
        // Show welcome message and start chat
        chatStarted = true;
        showWelcomeMessage();
        document.getElementById('chatFooter').style.display = 'block';
        
        // Focus on input field
        setTimeout(() => {
            document.getElementById('messageInput').focus();
        }, 100);
    } catch (error) {
        console.error('Error starting chat:', error);
        alert('Failed to start chat. Please try again.');
    }
}

function toggleMinimize() {
    const chatWindow = document.getElementById('chatWindow');
    const minimizeBtn = document.getElementById('minimizeBtn');
    
    isMinimized = !isMinimized;
    
    if (isMinimized) {
        chatWindow.classList.add('minimized');
        minimizeBtn.innerHTML = '<i class="fas fa-expand"></i>';
    } else {
        chatWindow.classList.remove('minimized');
        minimizeBtn.innerHTML = '<i class="fas fa-minus"></i>';
        
        // Focus on input field after restoring
        if (chatStarted) {
            setTimeout(() => {
                document.getElementById('messageInput').focus();
            }, 100);
        }
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function sendQuickMessage(message) {
    document.getElementById('messageInput').value = message;
    sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    input.value = '';
    input.style.height = 'auto';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send message to backend API
        const response = await fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: userInfo.name,
                message: message
            }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to send message');
        }
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Mark user message as seen
        markMessageAsSeen();
        
        // Add agent response
        addMessage(data.response, 'agent');
    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        
        // Show error message
        addMessage("Sorry, I'm having trouble responding right now. Please try again later.", 'agent');
    }
}

function addMessage(text, sender) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    
    // If it's an agent message, parse markdown
    if (sender === 'agent') {
        bubbleDiv.innerHTML = parseMarkdown(text);
    } else {
        bubbleDiv.textContent = text;
    }
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    if (sender === 'user') {
        timeDiv.innerHTML = `
            <span class="message-status">
                <i class="fas fa-check"></i>
            </span>
            <span>${timeString}</span>
        `;
        
        // Store message ID for updating status
        messageDiv.dataset.messageId = Date.now();
    } else {
        timeDiv.textContent = timeString;
        
        // Update unread count if chat is closed and message is from agent
        if (!chatOpen && sender === 'agent') {
            unreadCount++;
            updateChatBadge();
        }
    }
    
    contentDiv.appendChild(bubbleDiv);
    contentDiv.appendChild(timeDiv);
    messageDiv.appendChild(contentDiv);
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    messageHistory.push({ text, sender, timestamp: now });
}

function markMessageAsSeen() {
    const userMessages = document.querySelectorAll('.message.user');
    const lastUserMessage = userMessages[userMessages.length - 1];
    
    if (lastUserMessage) {
        const statusDiv = lastUserMessage.querySelector('.message-status');
        if (statusDiv) {
            statusDiv.innerHTML = '<i class="fas fa-check-double" style="color: var(--success-color);"></i>';
        }
    }
}

function showTypingIndicator() {
    document.getElementById('typingIndicator').classList.add('active');
    const messagesContainer = document.getElementById('chatMessages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTypingIndicator() {
    document.getElementById('typingIndicator').classList.remove('active');
}

function showFallbackMessage() {
    document.getElementById('fallbackMessage').classList.add('active');
    
    // Update agent status
    const statusDot = document.getElementById('agentStatus');
    const statusText = document.getElementById('agentStatusText');
    statusDot.classList.add('offline');
    statusText.textContent = 'Offline';
}

function updateChatBadge() {
    const chatBadge = document.getElementById('chatBadge');
    
    if (unreadCount > 0) {
        chatBadge.textContent = unreadCount > 9 ? '9+' : unreadCount;
        chatBadge.style.display = 'flex';
    } else {
        chatBadge.style.display = 'none';
    }
}

// Markdown parser function
function parseMarkdown(text) {
    // Convert headers
    text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    
    // Convert bold
    text = text.replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>');
    
    // Convert italic
    text = text.replace(/\*(.*)\*/gim, '<em>$1</em>');
    
    // Convert inline code
    text = text.replace(/`(.*)`/gim, '<code>$1</code>');
    
    // Convert code blocks
    text = text.replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>');
    
    // Convert links
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank">$1</a>');
    
    // Convert unordered lists
    text = text.replace(/^\* (.+)$/gim, '<ul><li>$1</li></ul>');
    text = text.replace(/<\/ul>\s*<ul>/g, '');
    
    // Convert ordered lists
    text = text.replace(/^\d+\. (.+)$/gim, '<ol><li>$1</li></ol>');
    text = text.replace(/<\/ol>\s*<ol>/g, '');
    
    // Convert blockquotes
    text = text.replace(/^> (.+)$/gim, '<blockquote>$1</blockquote>');
    
    // Convert line breaks
    text = text.replace(/\n/gim, '<br>');
    
    return text;
}

// For testing purposes - toggle API availability
function toggleApiAvailability() {
    isApiAvailable = !isApiAvailable;
    console.log('API availability:', isApiAvailable ? 'Online' : 'Offline');
    
    if (!isApiAvailable) {
        showFallbackMessage();
    } else {
        document.getElementById('fallbackMessage').classList.remove('active');
        
        // Reset agent status
        const statusDot = document.getElementById('agentStatus');
        const statusText = document.getElementById('agentStatusText');
        statusDot.classList.remove('offline');
        statusText.textContent = 'Online';
    }
}