var $messages = $('.messages-content'),
    d, h, m,
    i = 0;

//Initial greeting
var BotResponse = 'Hello, welcome to the financial news chatbot';

$(window).load(function() {
    $messages.mCustomScrollbar();
    setTimeout(function() {
        fakeMessage();
    }, 100);
});

function updateScrollbar() {
    $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
        scrollInertia: 10,
        timeout: 0
    });
}

function setDate() {
    d = new Date()
    if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
}

function insertMessage() {
    msg = $('.message-input').val();
    if ($.trim(msg) == '') {
        return false;
    }
    $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    $('.message-input').val(null);
    updateScrollbar();
    setTimeout(function() {
        fakeMessage();
    }, 1000 + (Math.random() * 20) * 100);
}

$('.message-submit').click(function() {
    insertMessage();
});

$(window).on('keydown', function(e) {
    if (e.which == 13) {
        insertMessage();
        return false;
    }
})

var Fake = [
    'Hi there, I\'m BATMAN and you?',
    'Do you wanna know My Secret Identity?',
    'Nice to meet you',
    'How are you?',
    'Not too bad, thanks',
    'What do you do?',
    'That\'s awesome',
    'Codepen is a nice place to stay',
    'I think you\'re a nice person',
    'Why do you think that?',
    'Can you explain?',
    'Anyway I\'ve gotta go now',
    'It was a pleasure chat with you',
    'Bye',
    ':)'
]

function getResponse(){
    message = $('.message-input').val()
    console.log(message)
    $.ajax({
        type: 'GET',
        url: 'http://127.0.0.1:8080/message/',
        data: {
          'message': msg
        },
        dataType: 'json',
        success: function(jsondata){
        console.log(jsondata['key']);
        BotResponse = jsondata['key']
        }
    })
}

function fakeMessage() {

    if ($('.message-input').val() != '') {
        return false;
    }
    $('<div class="message loading new"><figure class="avatar"><img src="img/ai-icon.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();
    getResponse();
    setTimeout(function() {
        $('.message.loading').remove();
        $('<div class="message new"><figure class="avatar"><img src="img/ai-icon.png" /></figure>' + BotResponse + '</div>').appendTo($('.mCSB_container')).addClass('new');
        console.log('New message')
        setDate();
        //getResponse();
        updateScrollbar();
        i++;
    }, (Math.random() * 20));

}
