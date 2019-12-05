from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.utils.crypto import get_random_string

from authentication.models import User
from django.contrib.auth.hashers import make_password

from django.contrib.auth import login, authenticate
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from authentication.tokens import account_activation_token
from django.core.mail import EmailMessage, send_mail


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')

    return render(request, 'auth/login.html')


def post_login(request):
    if not request.method == 'POST':
        messages.add_message(request, messages.ERROR, 'Only post method is allowed to perform this action')
        return redirect('login')

    # form = LoginForm(request.POST or None)
    # if not form.is_valid():
    #     messages.add_message(request, messages.ERROR, 'Email or password mismatch')
    #     return redirect('login', {'form': form})

    user = authenticate(
        request,
        email=request.POST.get('email'),
        password=request.POST.get('password'),
        is_active=True
    )
    if user is not None:
        login(request, user)

        if user.is_superuser:
            return redirect('/admin/')

        return redirect('index')
    else:
        messages.add_message(request, messages.ERROR, 'Email or password mismatch')
        return redirect('login')


def registration_view(request):
    return render(request, 'auth/register.html')


def registration(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['password_confirmation']

        if len(name) < 1:
            messages.error(request, 'Name field should be filled up!')
            return redirect('register')

        if password == confirm_password and len(password) >= 8:
            if User.objects.filter(email=email) or len(password) < 8:
                if User.objects.filter(email=email):
                    messages.error(request, 'Email being used!')
                    return redirect('register')
                if len(password) < 8:
                    messages.error(request, 'Password at least 8 character')
                    return redirect('register')
            else:
                user = User.objects.create(
                    name=name,
                    email=email,
                    password=make_password(password),
                )

                current_site = get_current_site(request)
                mail_subject = 'Activate your TaleantAI account.'
                message = render_to_string('emails/account_active.html', {
                    'user': user,
                    'domain': current_site.domain,
                    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                    'token': account_activation_token.make_token(user),
                })
                send_mail(mail_subject, message, 'no-reply@taleantai.com', [user.email])

                messages.add_message(request, messages.SUCCESS, 'Registration successful. Check you email to confirm.')
                return redirect('login')
        else:
            if password != confirm_password:
                messages.error(request, 'Opps! confirm password mismatch!')
                return redirect('register')
    else:
        messages.error(request, 'Only post method is allowed to perform this action')
        return redirect('register')


def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        return redirect('index')
    else:
        return HttpResponse('Activation link is invalid!')


def logout_user(request):
    logout(request)
    return redirect('/')


def password_forget(request):
    return render(request, 'auth/password_forget.html')


def password_reset(request):

    email = request.POST['email']


    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        messages.error(request, 'Invalid email!')
        return redirect('password_forget')

    user.token = get_random_string(length=32)
    user.save()
    current_site = get_current_site(request)
    mail_subject = 'Password reset email.'
    message = render_to_string('emails/password_reset.html', {
        'user': user,
        'domain': current_site.domain,
        'token': user.token,
    })
    send_mail(mail_subject, message, 'no-reply@taleantai.com', [user.email])

    return redirect('login')


def password_reset_view(self, token, *args, **kwargs):

    try:
        user = User.objects.get(token=token)
    except User.DoesNotExist:
        messages.error(self, 'Invalid url!')
        return redirect('password_forget')

    return render(self, 'auth/password_reset.html', {token: token})


def password_reset_post(request):
    
    token = request.POST['token']
    password = request.POST['password']
    confirm_password = request.POST['password_confirmation']
    try:
        user = User.objects.get(token=token)
        user.token = null = True

        if password == confirm_password:
            messages.error(request, 'Opps! confirm password mismatch!')
            return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
        if len(password) >= 8:
            messages.error(request, 'Password at least 8 character')
            return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
        
        user.password = make_password(password)
        messages.error(request, 'Password reset successful!')
        return redirect('login')
        
    except User.DoesNotExist:
        messages.error(request, 'Invalid token!')
        return redirect('login')
