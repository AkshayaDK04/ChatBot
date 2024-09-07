from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .nlp import greeting, response
from .forms import ChatForm
from datetime import datetime
import os

def signin(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        pass1 = request.POST.get('password')
        user = authenticate(request, username=uname, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('/')
        error = 'Username or password is incorrect'
        return render(request, 'login.html', {'error': error, 'username': uname})
    return render(request, 'login.html')

def signup(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')
        if pass1 != pass2:
            error = 'Passwords do not match'
            return render(request, 'signup.html', {'error': error, 'username': uname, 'email': email})
        if User.objects.filter(username=uname).exists():
            error = 'Username already exists'
            return render(request, 'signup.html', {'error': error, 'username': uname, 'email': email})
        if User.objects.filter(email=email).exists():
            error = 'Email already registered'
            return render(request, 'signup.html', {'error': error, 'username': uname, 'email': email})
        user = User.objects.create(username=uname, email=email)
        user.set_password(pass1)
        user.save()
        messages.success(request, 'Account created successfully. You can now log in.')
        return redirect('signin')
    return render(request, 'signup.html')
"""
def chatbot(request):
    if request.method == 'POST':
        user_query = request.POST['user_query']
        bot_response = response(user_query)  # Your logic to generate a bot response
        time = datetime.now().strftime('%H:%M')

        # Update chat history
        request.session['chat_history'].append({
            'user_query': user_query,
            'bot_response': bot_response,
            'time': time
        })

        return redirect('chat')  # Reload the page to display the updated chat

    return render(request, 'chat.html', {'chat_history': request.session.get('chat_history', [])})

"""
def chatbot(request):
    # Initialize chat history in session if it doesn't exist
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []

    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            user_message = form.cleaned_data['message'].strip()
            if greeting(user_message):
                bot_response = greeting(user_message)
            else:
                bot_response = response(user_message)
            
            # Append the new message and response to the chat history
            request.session['chat_history'].append({
                'user_query': user_message,
                'bot_response': bot_response
            })
            request.session.modified = True  # Ensure session is marked as modified

            # Clear the form for the next message
            form = ChatForm()
    else:
        form = ChatForm()
    
    # Pass the chat history to the template
    return render(request, 'chat.html', {
        'form': form,
        'chat_history': request.session['chat_history'],
    })
    

def home(request):
   # path=os.path.normpath('C:/Users/Smile/Documents/Chatbot_Project/college_chatbot/templates/home.html')
    return render(request, "home.html")
