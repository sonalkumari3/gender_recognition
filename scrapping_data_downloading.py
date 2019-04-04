###@Author Email: sonal.kumari1910@gmail.com

#### Importing Python libraries
import requests
import urllib.request
import os
from bs4 import BeautifulSoup

#url to download the data files"http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"
video_url =

response = requests.get(video_url)

if response.status_code == 200: ##Response code 200 is for valid url connection
    print('the response code is ',response + '\n')
    soup = BeautifulSoup(response.text, "html.parser")  ## parse the webpage
    untar_directory = 'untar_data1'  ##directory name to download the data
    if not os.path.exists(untar_directory):
        os.makedirs(untar_directory)

else:
    print('The given URL is not a valid url' + '\n')


#print(soup)

untar_directory= '8kHz_16bit' ##directory name to download the data
if not os.path.exists(untar_directory): os.makedirs(untar_directory)

for i in range(12,len(soup.findAll('a'))+1): #'a' tag finds all the links on the given webpage
    one_a_tag = soup.findAll('a')[i]  ##traverse through the links to download data files
    link = one_a_tag['href']
    download_url = video_url + link
    urllib.request.urlretrieve(download_url, untar_directory+'/'+link[link.find('/')+1:])


