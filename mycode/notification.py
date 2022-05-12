import requests

import getpass
import smtplib
from email.mime.text import MIMEText


GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587

class SMTPGmailNotifier:
    def __init__(self, address, password, smtp_host=GMAIL_SMTP_HOST,
                    smtp_port=GMAIL_SMTP_PORT, debug=False):
        """
        :param address: The email address to use (as all three of SMTP login 
                        username, email sender, and email recipient).
        :param password: The password to use for SMTP server login.
        :param smtp_host: Name of the SMTP server.
        :param smtp_port: Port of the SMTP server.
        :param debug: Set True to inhibit sending actual messages
        """
        print("Configuring SMTP Gmail Notifier...")
        self.address = address
        self.password = password
        self.host = smtp_host
        self.port = smtp_port
        self.debug = debug

    def notify(self, subject: str, text: str) -> None:
        """
        Send a self-email.

        :param subject: The email subject line.
        :param text: The email body text.
        """
        print("Sending an email to self...")
        print("From/To:", self.address)
        print("Subject:", subject)
        print("Message:", '"""', text, '"""', sep="\n")
        
        # make the email object
        msg = MIMEText(text)
        msg['To'] = self.address
        msg['From'] = self.address
        msg['Subject'] = subject

        # break early if in debug mode
        if self.debug:
            print("(Debug mode. Not sending)")
            return

        # log into the SMTP server to send it
        s = smtplib.SMTP(self.host, self.port)
        s.ehlo(); s.starttls()
        s.login(self.address, self.password)
        s.sendmail(self.address, [self.address], msg.as_string())
        s.quit()
        print("Sent!")

class TelegramBotNotifier:
    """
    Send notification using Telegram Bot API.

    Telegram bot can be created with @BotFather (https://t.me/botfather).
    """
    def __init__(self, token: str, chat: str) -> None:
        print("Configuring Telegram Bot Notifier...")
        self.entry_point = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat = chat

    def notify(self, subject: str, text: str) -> None:
        print("Sending Telegram Bot notification...")
        print("Message:", '"""', text, '"""', sep="\n")

        data = {
            "text": f"<b>{subject}</b>\n\n{text}",
            "chat_id": self.chat,
            "parse_mode": "html",
            "disable_web_page_preview": True
        }

        r = requests.post(self.entry_point, json=data)
        if r.status_code != 200:
            raise Exception(r.status_code, r.text)
        
        print("Sent!", r.text)

class MultiNotifier:
    def __init__(self, notifiers=None):
        if notifiers is not None:
            self.notifiers = notifiers
        else:
            self.notifiers = []

    def add_notifier(self, notifier):
        self.notifiers.append(notifier)

    def notify(self, subject: str, text: str) -> None:
        print("Triggering all notification methods...")
        problems = []
        nsuccess, nfail = 0, 0
        for notifier in self.notifiers:
            try:
                notifier.notify(subject, text)
                nsuccess += 1
            except Exception as e:
                problems.append((notifier, e))
                nfail += 1
        print(f"{nsuccess} notification methods triggered, {nfail} failed.")
        if problems != []:
            raise Exception("Some notification methods failed.", *problems)

NOTIFIER = MultiNotifier()
TELEGRAM_ACCESS_TOKEN = "737239960:AAEUK0C2Vb211gPRUWMffgrv1j6IBC0FuUw" 
TELEGRAM_DESTINATION  = 552384750 

GMAIL_ADDRESS  = '404notfxxkingfound@gmail.com'
GMAIL_PASSWORD = 'S13daowohao'
NOTIFIER.add_notifier(SMTPGmailNotifier(
    address=GMAIL_ADDRESS,
    password=GMAIL_PASSWORD))

# NOTIFIER.add_notifier(TelegramBotNotifier(
#    token=TELEGRAM_ACCESS_TOKEN,
#    chat=TELEGRAM_DESTINATION))