# Lab1 

## Installing and setting up the VM 

* (Quick) format your USB stick to NTFS

* Download in your tmp directory: 

[mlvm](https://docs.google.com/uc?id=0B_kDfEzMuWD6ZGJFU1VfeEY3TnM&export=download)

* Copy the VM to the USB stick

* Start the VM 
	* Double click on mlvm.vbox
	* An error might pop up
	* Press change settings and change the network adapter to another value, and then back to NAT

-------

## Login to IRC [https://kiwiirc.com/client](https://kiwiirc.com/client)
	

* Unrar and move "mlvm" directory to usb stick



* Login using mlvm/mlvm
	* Click on the vm window
	* (To exit vm mode press ctrl+alt)
	* Type "sudo reboot" to reboot the machine
	* Find your local ip address (type "ifconfig")
	* Open an explore window and type "\\\\mlvm\\mlvm"
	* Enter your credentials (mlvm/mlvm)

* Extra package installation
	* sudo apt-get install enchant
	* sudo pip install sopel 

## Running a Sopel bot

* type `sopel' AND ENTER YOUR OWN NICKNAME

I can't seem to find the configuration file, so let's generate it!

Please answer the following questions to create your configuration file:

Enter the nickname for your bot. [Sopel] ssamotbot
Enter the server to connect to. [irc.dftba.net] irc.freenode.net
Should the bot connect with SSL? (y/n) [n] y
Enter the port to connect on. [6697]
Enter your own IRC name (or that of the bot's owner) ssamotbot
Enter the channels to connect to at startup, separated by commas. []
? #ce888
?
Would you like to see if there are any modules that need configuring (y/n)? [n]
Config file written sucessfully!


Once you have reached this point, press ctrl+c to exit (once the bot is connected)

cd .sopel
nano default.cfg

add to the end "extra = ."

Save and exit

## Sopel first module

from sopel import module

@module.rule('')
def hi(bot, trigger):
    bot.say('Hi, ' + trigger.nick)


## Installing nltk!

sudo pip install nltk
sudo python -m nltk.downloader -d /usr/local/share/nltk_data punkt
sudo python -m nltk.downloader -d /usr/local/share/nltk_data wordnet
sudo python -m nltk.downloader -d /usr/local/share/nltk_data averaged_perceptron_tagger



