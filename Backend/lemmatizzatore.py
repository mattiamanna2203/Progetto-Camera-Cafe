import spacy                    
nlp = spacy.load('it_core_news_sm') 

from nltk.stem import SnowballStemmer 
stemmer_snowball = SnowballStemmer('italian') 

import unicodedata 

import re

def cleaning_text(text : str) -> str:
   """ Funzione per la prima pulizia del testo, rimuovere punteggiatura, spazi superflui e normalizzazione"""

   # Rimuovere segni di punteggiatura.
   # In particolare rimuove: , . ? ! # @ & $ £ % < > | = : ; + * _ / -  " ' ^ ( ) { } [ ] \n \  
   text = re.sub("[, \. ? ! # @ & $ £ % < > | = : ; \+ \* _ / \-  \"  \^ ( ) \{ \} \[ \] \n \\\  ]"," ",text)
   #text =  re.sub(r'[^a-zA-Z0-9\s]', '', text) # Troppo aggressivo 

   # Rimuovere spazi in eccesso 
   text = text.strip() # Rimuovere spazi iniziali e finali
   text = re.sub( r'\s{2,}'," ",text) # Rimuovere gli spazi all'interno della frase più lunghi di un carattere


   return unicodedata.normalize('NFC',text) 



def lemmatizzazione(text : str) -> list:
   """Lemmatizzazione del testo:
         - Portare tutti i verbi all'infinito 
         - Tutti i nomi al maschile
         - articoli ricodificati nella forma maschile
         - Preposizioni ricodificati 

      Altre modifiche al testo:
         - Rimozione congiunzioni 
         - Rimozione preposizioni
         - Portare tutto ciò che non è un nome in minuscolo 
         - Ottenere una lista dei personaggi presenti

   Nota: Questa funzione dovrà essere applicata anche alle query degli utenti
   """

   # Cleaning del testo, richiamare la funzione definita in precedenza
   text = cleaning_text(text)


   # Nomi dei personaggi e altri modi con i quali vengono identificati, per ora solo definiti e non utilizzati
   nomi_personaggi = {"Luca":["Luca","Luca Nervi","Nervi"],
                      "Paolo":["Paolo","Paolo Bitta","Bitta"],
                     "Andrea":["Andrea","Andrea Pellegrino","Pellegrino"],
                     "Ilaria":["Ilaria","Ilaria Tanadale","Tanadale","direttrice marketing"],
                     "Alex":["Alessandra","Alessandra Costa","Costa","Alex"],
                     "Gaia":["Gaia","Gaia De Bernardi","Gaia DeBernardi","DeBernardi","De Bernardi"],
                     "Silvano":["Silvano","Silvano Rogi","Rogi"],
                     "De Marinis":["Augusto","Augusto De Marinis","Augusto DeMarinis","De Marinis","DeMarinis","Direttore","Marinis","Demarinis"],
                     "Olmo":["Olmo","Olmo Ghesizzi","Ghesizzi"],
                     "Geller":["Geller","Guido Geller","Dottor Geller","Guido"],
                     "Patty":["Patty","Patrizia","Patty Dimporzano","Patrizia Dimporzano","Patty D'Imporzano","Patrizia D' Imporzano"],
                     "Lucrezia":["Lucrezia","Lucrezia Orsini","Dottoressa Orsini","Orsini"],
                     "Maria Eleonora":["Maria","Eleonora","Maria Eleonora Bau","Dottoressa Bau"],
                     "Anna":["Anna","Anna Murazzi","Murazzi"],
                     "Emma":["Emma","Emma Missale"],
                     "Giovanna":["Giovanna","Giovanna Caleffi"],
                     "Presidente":["Presidente","Il Presidente"],
                     "Pippo":["Pippo","Giuseppe Locascio","Giuseppe Lo Cascio"],
                     "Anselmo":["Anselmo","Anselmo Pedone","Pedone"],
                     "Gloria":["Gloria","Gloria Spallone"],
                     "Jessica":["Jessica"],
                     "Lello":["Lello"],
                     "Valeria":["Valeria"],
                     "Brad":["Brad"],
                     "Jonathan":["Jonathan"],
                     "Signora Bollini":["Signora Bollini"],
                     "Mamma di Silvano":["Mamma di Silvano"],
                     "Wanda":["Wanda","Wanda Sordi"],
                     "Vittorio" : ["Vittorio","Vittorio Ubbiali","guardia","volpe di fuoco"],   
                     "Carminati":["Michele","Michele Carminati","Carminati"],
                     "Caterina":["Caterina"],
                     "Pooh" :["Pooh","Roby","Roby Facchinetti","Facchinetti","Valerio Negrini","Negrini","Dodi Battaglia","Dodi","Red Canzian","Canzian","Stefano D'Orazio","D'Orazio","Poo","Pooo"],
                     "Riccardo Fogli":["Riccardo Fogli"],
                     "Roberto" :["Roberto"]
                  }

   # Lemmatizazione tramite spaCy
   lemmatizzazione = nlp(text) # Si avrà un oggetto che darà per ogni parola la sua forma nel testo ed il suo lemma (ovvero se è un verbo la forma all'infinito ecc...)

   # Lista che spacy sembra non identificare ma che sono da rimuovere
   parole_da_rimuovere = ["eh","ah","oh","non","beh","no","si"]

   # Prendere solo la versione LEMMA dal risultato della lematizzazione escludendo punteggiatura, congiunzioni ed articoli
   lista_parole_lemmatizzate = [ parola.lemma_ for parola in lemmatizzazione 
                                 if parola.pos_ not in ['DET', 'CCONJ','ADP','PART',"SYM","SPACE","X","INTJ","SCONJ","PRON"] # Escludere se la parola fa parte di: 
                                    and 
                                  parola.lemma_  not in parole_da_rimuovere
                               ]


   # Estrarre tutti i nomi propri, non è molto buona sembra prendere solo le parole maiuscole
   #lista_nomi = [ parola.lemma_ for parola in lemmatizzazione if parola.pos_  in ['PROPN']]
   lista_nomi = [ str(parola.lemma_).title() for parola in lemmatizzazione if parola.pos_  in ["NOUN","PROPN"] ]





   # Identificare i personaggi
   personaggi_presenti = [] # Inizializzare una lista  che conterrà tutti i personaggi nominati nel testo

   # Iterare sui nomi dei personaggi
   for personaggio, nomi_assegnati_al_personaggio in nomi_personaggi.items():

      # Iterare sui nomi propri identificati  tramite spacy
      for parola in lista_nomi:

         # Se un nome proprio identificato corrisponde con un nome di un personaggio:
         if str(parola) in nomi_assegnati_al_personaggio:

            personaggi_presenti.append(personaggio) # Aggiungere il personaggio alla lista di quelli ritrovati
            lista_nomi.remove(parola) # Rimuovere quella parola dalla lista nomi così da non considerla più
            break  # interrompere visto che il nome del personaggio è stato trovato ed proseguire con il successivo







   # Stemmare tutti i Lemmi (procedura che per ora non viene applicata)
   #lista_parole_stemmate = [stemmer_snowball.stem(lemma) for lemma  in lista_parole_lemmatizzate]


   # Ora che i personaggi sono stati identificati procedere rendere tutto minuscolo
   lista_parole_lemmatizzate  =list(map(lambda parola: parola.lower(), lista_parole_lemmatizzate))


   # Rimuovere tutti i duplicati? Non penso sia utile proprio per come tf idf è costruito. per questo non è attivo
   #lista_parole_lemmatizzate = list(set(lista_parole_lemmatizzate))

   return lista_parole_lemmatizzate,personaggi_presenti

