import arduino
import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy
#from scipy import signal

from pubnub.pubnub import PubNub, SubscribeListener, SubscribeCallback, PNStatusCategory
from pubnub.pnconfiguration import PNConfiguration
from pubnub.exceptions import PubNubException
import pubnub
import time 
    
pnconf = PNConfiguration()                         # create pubnub_configuration_object

pnconf.publish_key = 'your pubnbe publish_key'           # set pubnub publish_key
pnconf.subscribe_key = 'you pubnub subscribe_key'       # set pubnub subscibe_key
pnconf.uuid = 'K'

pubnub = PubNub(pnconf)                            # create pubnub_object using pubnub_configuration_object

channel = 'Your Channel Name'                                     # provide pubnub channel_name
data = {                                           # data to be published
'message': 'hi'
}

data2 = 1000

my_listener = SubscribeListener()                  # create listner_object to read the msg from the Broker/Server
pubnub.add_listener(my_listener)                   # add listner_object to pubnub_object to subscribe it

pubnub.subscribe().channels(channel).execute()     # subscribe the channel (Runs in background)
my_listener.wait_for_connect()                     # wait for the listener_obj to connect to the Broker.Channel

print('connected')                                     # print confirmation msg
pubnub.publish().channel(channel).message(data).sync() # publish the data to the mentioned channel





# filter top n frequencies
n_fabs_filter = 1

# Sampling parameters
sample_rate = 1000
sampling_interval = 50 # miliseconds interval for sampling
sample_size = 10
# sampling_timespace
dt = sampling_interval / sample_size
delay_time = 10 # delay between each sampling

# Create the I2C bus with a fast frequency
# NOTE: Your device may not respect the frequency setting
#       Raspberry Pis must change this in /boot/config.txt
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c, address=0x48)
chan0 = AnalogIn(ads, ADS.P0)

# First ADC channel read in continuous mode configures device
# and waits 2 conversion cycles
_ = chan0.value
noise_values = np.zeros((1,sample_size))
i=0
while (i<1):
   i = i + 1
   samples = [0]*sample_size
   t = [0]*sample_size
   start_millis = int(time.time() *1000)
   for j in range(0,sample_size):
       current_millis = start_millis
       while ((current_millis-start_millis)<dt*(j+1)):
           current_millis = int(time.time() *1000)
       t[j] = current_millis-start_millis
       samples[j] = chan0.value
   

   a = np.array(samples)
   noise_values = np.concatenate((noise_values,[a]), axis = 0)

   
   # Check Stackoverflow website : How to convert from frequency to time domain in python
   
   
   # Perform FFT using scipy.fftpack.fftfreq(n, d=1.0)
   # n: window length, d:sample spacing(inverse of the sampling rate)
   # the returned float array f contains the frequency bin
   # centres in cycles per unit of the sample spacing (with zero at start)
   #freq = np.fft.fftfreq(sample_size,dt) # array of lengthn containing the sample frequencies
   freq = (1/(dt*sample_size) * np.arange(sample_size))
   fhat = np.fft.fft(samples) # compute the FFT
   #fabs = np.abs(fhat[:sample_size])
   fabs = np.abs(fhat * np.conj(fhat) / sample_size)
   #L = np.arange(1,np.floor(sample_size/2), dtype=np.int8)
   
   # Filter FFT values
   sorted_fabs_index = np.argsort(fabs)
   sorted_fabs = fabs[sorted_fabs_index]
   print(sorted_fabs)
   max_fabs = sorted_fabs[-n_fabs_filter :]
   filtered_fabs_indices = (fabs<max_fabs[n_fabs_filter-1])
   filtered_fhat_array = filtered_fabs_indices * fhat
   fabs_filtered = fabs * filtered_fabs_indices
   #fhat_filtered = filtered_fabs_indices * fhat
   #fabs_filtered_sample = np.abs(fhat_filtered[:sample_size])
   #fft_filtered_sample = np.fft.ifft(fabs_filtered)
   #fabs_filtered_sample = np.abs(fft_filtered_sample)
   filtered_sample = np.abs(np.fft.ifft(filtered_fhat_array))
   
   
   output_signal = (max_fabs[0])#+max_fabs[1]+max_fabs[2])/3
   data2 = "Noise=" + str(output_signal) + ", Temp=10, Humidity=20"
   pubnub.publish().channel(channel).message(data2).sync()
   
   
   #freq = (1/(dt*n))*np.arange(n) # frequency array
   #idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32) # first half index

# Plot figures in fourier domain
fig1, axs = plt.subplots(2,1)
plt.sca(axs[0])
plt.plot(freq, fabs, '--b', label = 'Operation noise in frequency domain')
plt.plot(freq,fabs_filtered, 'g', label = 'Clean signal in frequency domain')
plt.xlabel("Frequency (HZ)")
plt.ylabel("frequency amplitude")
plt.title("Filtered amplitude to eliminate major harmonics")

plt.sca(axs[1])
#plt.plot(t, samples, '--c', linewidth=1.5, label='Operation noise in time domain')
plt.plot(t, filtered_sample, 'k', linewidth=2, label='Clean signal in time domain')
plt.xlabel("Time (ms)")
plt.ylabel('Signal value')
plt.title("Signal in time domain")
plt.legend()


# Plot figures in time domain
#t = np.arange(0, sampling_interval, dt)
#fig, axs = plt.subplots(2, 1)

#plt.sca(axs[0])
#plt.plot(t, samples, color='c', linewidth=1.5, label='Noisy')
#plt.plot(t, samples, color='k', linewidth=2, label='Clean')
#plt.xlim(t[0], t[-1])
##plt.xlabel('t axis')
#plt.ylabel('Vals')
#plt.legend()

#plt.sca(axs[1])
#plt.plot(freq[idxs_half], psd[idxs_half], color='c', linewidth=2, label='PSD Noisy')
#plt.xlim(freq[idxs_half[0]], freq[idxs_half[-1]])
#plt.xlabel('t axis')
#plt.ylabel('Vals')
#plt.legend()

plt.tight_layout()
plt.show()
