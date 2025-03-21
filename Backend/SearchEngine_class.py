import pandas as pd
# Mostra tutte le colonne
pd.set_option('display.max_columns',6)

# Mostra tutto il contenuto delle colonne
pd.set_option('display.max_colwidth', 200)

from  lemmatizzatore import lemmatizzazione
import json
import re
from alive_progress import alive_bar
from typing import Union, List # per specificare campi multipli nelll'input funzione
import numpy as np
import math

def aliveit(progress_bar,loop_length,progresso=0):
   """Funzione per aggiornare manualmente una barra di progresso alive"""
   progresso += 1 
   progress_bar(progresso / loop_length)
   return progresso

class SearchEngine:
   """Classe per la search engine di camera cafe, siccome le funzioni da definire erano molto 
      ed i dati sempre  gli stessi si è creata una classe per cercare di semplificare il tutto.
      è anche un discreto esercizio di stile per imparare a programmare decentemente.
   """
   def __init__(self,path : str):
      if  not isinstance(path,str):
         raise TypeError("Specificare il percorso file della directory nella quale sono i dati")

      self.path = path
      self.colonne_per_output = ['season', 'episodio', 'titolo', 'trama']
      try:
         # Ripasso:
         # Vocabulary:  Indicato per ogni parola il numero di volte che compare (conteggio su tutti i documenti)
         # Word Dict: Assegnare ad ogni parola un numero identificativo
         # Inverted Index: per ogni parola indicata con il numero specificato nel word_dict vengono elencati i documenti nella quale compare.
         # Tf - Idf: Indice per capire la rilevenza di una parola in un documento, se appare raramente ogni volta che appare in un documento avrà importanza maggiore rispetto a parole che compaiono poco.


         loop_length = 7#14
         with alive_bar(loop_length, manual=True,force_tty=True, title="Caricamento dati: ") as bar:
         
            # Importare il database degli episodi, contiene titoli delle puntate, trama, copione...
            self.df = pd.read_parquet(f"{path}/dati_puliti_aggiornati.parquet", engine="pyarrow")
            progresso = aliveit(progress_bar=bar,loop_length=loop_length)

            self.tf_idf_dataframe_title_trama = pd.read_parquet(f"{path}/tfidf_index_title_trama.parquet", engine="pyarrow")
            progresso = aliveit(progress_bar=bar,loop_length=loop_length,progresso=progresso)


            with open(f"{path}/word_dict_title_trama.json", 'r') as wd:
               self.word_dict_title_trama  = json.load(wd)
            progresso = aliveit(progress_bar=bar,loop_length=loop_length,progresso=progresso)


            ## Import inverted_indexes
            with open(f"{path}/inverted_idx_title_trama.json", 'r') as inv_idx:
               self.inverted_idx_title_trama = json.load(inv_idx)
            progresso = aliveit(progress_bar=bar,loop_length=loop_length,progresso=progresso)




   
            self.tf_idf_dataframe_all= pd.read_parquet(f"{path}/tfidf_index_all.parquet", engine="pyarrow")
            progresso = aliveit(progress_bar=bar,loop_length=loop_length,progresso=progresso)




            with open(f"{path}/word_dict_all.json", 'r') as wd:
               self.word_dict_all  = json.load(wd)
            progresso = aliveit(progress_bar=bar,loop_length=loop_length,progresso=progresso)

            ## Import inverted_indexes
            with open(f"{path}/inverted_idx_all.json", 'r') as inv_idx:
               self.inverted_idx_all = json.load(inv_idx)
            progresso = aliveit(progress_bar=bar,loop_length=loop_length,progresso=progresso)

            self.nomi_personaggi = {"Luca":["Luca","Luca Nervi","Nervi"],
                     "Paolo":["Paolo","Paolo Bitta","Bitta"],
                  "Andrea":["Andrea","Andrea Pellegrino","Pellegrino"],
                  "Ilaria":["Ilaria","Ilaria Tanadale","Tanadale","direttrice marketing"],
                  "Alex":["Alessandra","Alessandra Costa","Costa","Alex"],
                  "Gaia":["Gaia","Gaia De Bernardi","Gaia DeBernardi","DeBernardi","De Bernardi"],
                  "Silvano":["Silvano","Silvano Rogi","Rogi"],
                  "De Marinis":["Augusto","Augusto De Marinis","Augusto DeMarinis","De Marinis","DeMarinis","Direttore"],
                  "Olmo":["Olmo","Olmo Ghesizzi","Ghesizzi"],
                  "Geller":["Geller","Guido Geller","Dottor Geller","Guido"],
                  "Patty":["Patty","Patrizia","Patty Dimporzano","Patrizia Dimporzano","Patty D'Imporzano","Patrizia D' Imporzano"],
                  "Lucrezia":["Lucrezia","Lucrezia Orsini","Dottoressa Orsini","Orsini"],
                  "Maria Eleonora":["Maria","Eleonora","Maria Eleonora Bau","Dottoressa Bau"],
                  "Anna":["Anna","Anna Murazzi","Murazzi"],
                  "Emma":["Emma","Emma Missale"],
                  "Giovanna":["Giovanna","Giovanna Caleffi"],
                  "Presidente":["Presidente","Il presidente"],
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
          
      except:
         raise ValueError("Errore nell'importazione dati, controllare il path e che tutti i file richiesti esistano")




   def __evaluateTFIDF__ (self,word_dict,inverted_idx,lemmatized_query,lemmatized_query_original):
      dizionaro_tfidf = {}
      n_documents = self.df.shape[0]

      for word in lemmatized_query:

         # Contare occorrenza della i-esima parola nella query lemmatizzata originale, così da avere delle ripetizioni
         term_occurencies_in_document=" ".join(lemmatized_query_original).count(word) 


         # Contare le parole nella query lemmatizzata originale così da avere il giusto numero di parole
         terms_in_document=len(lemmatized_query_original) 


         # Ricavare la TF (Term Frequency), ovvero la frequenza relativa di una parola nel documento in questione
         # Quante volte la parola appare nel documento / quante parole ci sono nel documento

         tf=term_occurencies_in_document/terms_in_document


         # IDF (invers document frequency), proporzione dei documenti nei quali appare la parola
         # logaritmo di: numero di documenti totali / (numero documenti nei quali appare la parola +1), +1 per evitare divisioni per 0
         word_id = word_dict[word] # prendere l'id della parola
         documents_with_the_word = inverted_idx[str(word_id)] # tramite id della parola ottenere lista dei documenti nei quali appare
         n_documents_with_the_word = len(documents_with_the_word) # contare il numero di documenti ove appare la parola

         # Formula  e calcolo per l'idf
         idf = math.log(n_documents/(n_documents_with_the_word+1))
         
         # Formula e calcolo per tf*idf
         tf_idf = tf*idf

         # Salvare il tf-idf per la parola 
         dizionaro_tfidf[word] = tf_idf

      tf_idf_query_dataframe = pd.DataFrame.from_dict(dizionaro_tfidf,orient="index").T
      return tf_idf_query_dataframe


   def select_by_parameters(  self,
                              season: Union[int, List[int]],
                              personaggi_apparsi : list,
                              gueststar_apparse : list,
                              output_length: int = 20) -> pd.DataFrame():
      """Estrarre utilizzando solo: 
         - Stagione
         - Personaggi apparsi
         - Guest stars
      """

      df = self.df
      if season != None:
         # Siccome season può essere una lista o un numero intero.
         # Ma le righe successive utilizzano sintassi per le liste, mettere il numero intero in una lista
         if isinstance(season, int):
            season = [season]
         # Restringere il dataframe df alle stagioni selezionate
         df = df[df.season.isin(season)]

      if personaggi_apparsi != None:
         df = df[(df[personaggi_apparsi] != 0).all(axis=1)]
   

      if gueststar_apparse != None:
         df = df[(df[gueststar_apparse] != 0).all(axis=1)]
      
      df = df[self.colonne_per_output]

      return df.head(output_length)



   # personaggi_apparsi -> da implementare dovranno essere sull'html come le guest star (tendina selezionabile)
   # gueststar_apparse  -> da implementare 
   def all_matching( self,
                     query_str : str,
                     campi_ricerca : str  = "ALL",
                     season: Union[int, List[int]] = None, 
                     output_length: int = 20,
                     personaggi_apparsi: list = None,
                     gueststar_apparse: list = None) -> pd.DataFrame():
      """ ALL matching, trovare i documenti che contengono tutte le parole della query passata in input.
         Sfruttando gli inverted indexes.
          Input:
            - query_str -> parole da ricercare 
            - campi_ricerca -> su cosa ricercare:
               * 'ALL' (default) ricerca su: titoli, trame, copioni
               * 'title_trama' ricerca su: titoli, trame
            - season (default None) -> stagione o stagioni sulla quale restringere la ricerca, di default non è attivata
            - output_length (default None) -> Numero massimo di righe in output. 
                                              Vengono restituite le prime output_length righe 
            - personaggi_apparsi (default None) -> Restringere il campo a quelle puntate ove appaioni determinati personaggi
            - gueststar_apparse  (default None) -> Restringere il campo a quelle puntate ove appaioni determinate guest star
         Output:
            - Dataframe con le informazioni delle puntate che contengono tutte le parole nella query passata in input.
      """
      #print(f"query_str: '{query_str}',   all_match={all_match}")
      # region Controllare parametri input 
      # Check sulla query, deve essere una stringa
      if not isinstance(query_str,str):
         raise TypeError("Specificare con una stringa")

      # Check sul campo di ricerca, deve essere una stringa
      if  not isinstance(campi_ricerca,str):
         raise TypeError("Specificare con una stringa  su quali campi effettuare la ricerca")

      # Check sul campo di ricerca, deve essere uno dei 3
      if campi_ricerca not in ["ALL","title_trama"]:
         raise ValueError("Specificare una campo di ricerca valido")

      # Check sul numero massimo di output desiderati, output_length is not None aggiunto perchè di default è None
      if not isinstance(output_length, (int)):
         raise TypeError("Specificare una lunghezza massima per l'output valida, inserire un numero intero")

      # Check sulle stagioni sulle quali restringere la ricerca

      # Controllare che season sia un intero o una lista,  season is not None aggiunto perchè di default è None
      if  season is not None and  not isinstance(season, (int,list)):
         raise TypeError("Inserire un valore valido per la stagione, specificare un numero intero o una lista di numeri interi")

      # Se season è un intero deve essere uguale a  1,2,3,4,5,6, se non lo è c'è un errore
      if  (isinstance(season, int)) & (season not in [1,2,3,4,5,6]):
         raise ValueError("Inserire una stagione valida, le stagioni vanno da 1 a 6")

      # Se è una lista controllare che sia una lista di interi e controllare che i numeri siano 1,2,3,4,5,6
      if  isinstance(season, list):
         for season_number in season:
            if not isinstance(season_number, int):
               raise TypeError("Inserire un valore valido per la stagione, specificare un numero intero")
            if season_number not in [1,2,3,4,5,6]:
               raise ValueError("Inserire una stagione valida, le stagioni vanno da 1 a 6")


      # Controllare che le guest star siano fornite in una lista
      if gueststar_apparse is not None and  not isinstance(gueststar_apparse,list):
         raise TypeError("Inserire una lista di guest star")

      # Controllare che le guest star nella lista siano stringhe
      if  isinstance(gueststar_apparse, list):
         for guest_star in gueststar_apparse:
            if not isinstance(guest_star, str):
               raise TypeError("Inserire un valore valido per la guest star, specificare una stringa")


      # Controllare che le guest star siano fornite in una lista
      if personaggi_apparsi is not None and  not isinstance(personaggi_apparsi,(list)):
         raise TypeError("Inserire una lista di personaggi")

      # Controllare che le guest star nella lista siano stringhe
      if  isinstance(personaggi_apparsi, list):
         for personaggio in personaggi_apparsi:
            if not isinstance(personaggio, str):
               raise TypeError("Inserire un valore valido per il personaggio, specificare una stringa")

      # endregion


      df = self.df
      # Lista delle colonne da estrarre dal df da mostrare all'utente come output della ricerca
      colonne_per_output = self.colonne_per_output
      
      # region Cambiamenti a seconda del campo di ricerca selezionato
      if campi_ricerca == "ALL":

         # Prendere il word dict, identificativo numerico per ogni parola
         word_dict = self.word_dict_all

         # Inverted Index: per ogni parola indicata con il numero specificato nel word_dict vengono elencati i documenti nella quale compare.
         inverted_idx = self.inverted_idx_all

      elif campi_ricerca == "title_trama":
         # Stesse operazioni dell'if, con alcuni accorgimenti specifici per questo caso
         word_dict = self.word_dict_title_trama
         inverted_idx = self.inverted_idx_title_trama


   
      # endregion

      # Se non è stata inserita alcuna parola nella query restituire tutti gli episodi con i personaggi, stagione e guest star selezionate
      # Per capire se non è stata inserita alcuna parola rimuovere tutti gli spazi, se la stringa è uguale a ""
      # significa che non c'è alcuna parola
      query_str_no_spaces = re.sub( r'\s{1,}',"",query_str) # In questo modo "   " e " " e "" risultano uguali
      if query_str_no_spaces == "":
         output = self.select_by_parameters(season,personaggi_apparsi,gueststar_apparse,output_length)

         # Se l'output ha 0 righe significa che non ci sono match
         if output.shape[0] == 0:
            #raise Exception("Non ci sono puntate che corrispondono ai criteri di ricerca")
            print("Nessuna puntata ritrovata che abbia tutte le caratteristiche specificate")
            return None

         return output[colonne_per_output]


      # region Ricerca delle puntate che soddisfano la query
      # Lemmatizzare la query in modo da aver corrispondenza con i word dict e gli inverted indexes Visto che sono tutti lemmatizzati 
      lemmatized_query, output_inutile = lemmatizzazione(query_str)    

      # Se si vuole che tutte le parole della query vengano ritrovate 
      # non utilizzare lo statement "if parola in word_dict_trascrizione.keys()"
      # che permetteva di evitare i keyerror
 
      # Siccome la lemmatizzazione considera differente la stessa parola con iniziale maiuscola o minuscola
      # Per ampliare la ricerca utilizzare .title() che mette tutte le parole con iniziale maiuscola
      seconda_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.title())    
   
      # fare la stessa cosa mettendole tutte minuscole
      terza_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.lower())  

      # fare la stessa cosa mettendole tutte maiuscole
      quarta_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.upper())  

      # fare la stessa cosa mettendole tutte maiuscole
      quinta_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.capitalize())  

      opzioni_lemmatizzazione = [   lemmatized_query,
                                    seconda_opzione_lemmatizzazione,
                                    terza_opzione_lemmatizzazione,
                                    quarta_opzione_lemmatizzazione,
                                    quinta_opzione_lemmatizzazione
                                 ]
      # Iterare su tutte le opzioni di lemmatizzazione
      n_word_retrieved_best_option = 0
      indici_episodi = []
      for opzione_lemmatizzazione in opzioni_lemmatizzazione:
         
         # Cercare tutte le parole della lemmatized_query in word_dict, per evitare errori prendere solo quelle esistenti
         indici_episodi_iter = [word_dict[parola] for parola in  opzione_lemmatizzazione if parola in word_dict.keys()] 
         if len(indici_episodi_iter) > n_word_retrieved_best_option:
            indici_episodi = indici_episodi_iter
            n_word_retrieved_best_option =  len(indici_episodi_iter)

            

      # Se indici_episodi ha len 0 vuol dire che non ci sono puntate che corrispondono ai criteri di ricerca
      # così si interrompe la funzione e si restituisce che nessuna puntata è stata trovata
      if len(indici_episodi) == 0:
         print("Non ci sono episodi contenenti le keyword indicate")
         return None

 
      # Prendere la lista degli episodi nei quali ogni parola compare (una lista di episodi per ogni parola)
      puntate = [set(inverted_idx[str(indice)]) for indice in indici_episodi]

      # Cercare le puntate comuni a tutte le parole, si farà l'intersezione per ogni lista 

      # La  lista degli episodi comuni a tutti  inizializzata con la lista episodi della prima parola
      puntate_comuni = puntate[0]

      # Iterare sul resto delle liste degli episodi per le altre parole (si parte dalla seconda parola)
      for i in range(1,len(puntate)):

         # Prendere gli episodi in comune
         puntate_comuni = puntate_comuni.intersection(puntate[i])

      # Trasformare il set in una lista così da ottenere degli indici utili per identificare le puntate comuni a tutte le parole del dataset principale (df)
      index_puntate_comuni = list(puntate_comuni)
      # endregion

      

      df = df.loc[index_puntate_comuni]
      if season != None:
         # Siccome season può essere una lista o un numero intero.
         # Ma le righe successive utilizzano sintassi per le liste, mettere il numero intero in una lista
         if isinstance(season, int):
            season = [season]

         # Restringere il dataframe df alle stagioni selezionate
         df = df[df.season.isin(season)]

      if personaggi_apparsi != None:
         df = df[(df[personaggi_apparsi] != 0).all(axis=1)]
   

      if gueststar_apparse != None:
         df = df[(df[gueststar_apparse] != 0).all(axis=1)]


      # Se l'output ha 0 righe significa che non ci sono match
      if df.shape[0] == 0:
         #raise Exception("Non ci sono puntate che corrispondono ai criteri di ricerca")
         print("Nessuna puntata ritrovata che abbia in comune tutte le keyword selezionate")
         return None

      # endregion

      df = df[colonne_per_output]

      # Se viene specificata una lunghezza per l'output utilizzarla, si prendono i primi k, k=output_length
      pd.set_option('display.max_rows', output_length)
      return df.head(output_length)





   def ranking_matching( self,
                     query_str : str,
                     campi_ricerca : str  = "ALL",
                     season: Union[int, List[int]] = None, 
                     output_length: int = 20,
                     personaggi_apparsi: list = None,
                     gueststar_apparse: list = None) -> pd.DataFrame():
      """ Funzione per trovare le puntate più attinenti alla stringa che si sta ricercando.
         Sfrutta la cosine similarity sui tf-idf  per trovare i risultati migliori.
            - query_str -> parole da ricercare 
            - campi_ricerca -> su cosa ricercare:
               * 'ALL' (default) ricerca su: titoli, trame, copioni
               * 'title_trama' ricerca su: titoli, trame
            - season (default None) -> stagione o stagioni sulla quale restringere la ricerca, di default non è attivata
            - output_length (default 20) -> Numero massimo di righe in output. 
                                              Vengono restituite le prime output_length righe 
            - personaggi_apparsi (default None) -> Restringere il campo a quelle puntate ove appaioni determinati personaggi
            - gueststar_apparse  (default None) -> Restringere il campo a quelle puntate ove appaioni determinate guest star
         Output:
            - Dataframe con le informazioni delle puntate più attinenti
      """
 

      # region Controllare parametri input 
      # Check sulla query, deve essere una stringa
      if not isinstance(query_str,str):
         raise TypeError("Specificare con una stringa")

      # Check sul campo di ricerca, deve essere una stringa
      if  not isinstance(campi_ricerca,str):
         raise TypeError("Specificare con una stringa  su quali campi effettuare la ricerca")

      # Check sul campo di ricerca, deve essere uno dei 3
      if campi_ricerca not in ["ALL","title_trama","copione"]:
         raise ValueError("Specificare una campo di ricerca valido")

      # Check sul numero massimo di output desiderati, output_length is not None aggiunto perchè di default è None
      if not isinstance(output_length, (int)):
         raise TypeError("Specificare una lunghezza massima per l'output valida, inserire un numero intero")

      # Check sulle stagioni sulle quali restringere la ricerca

      # Controllare che season sia un intero o una lista,  season is not None aggiunto perchè di default è None
      if  season is not None and  not isinstance(season, (int,list)):
         raise TypeError("Inserire un valore valido per la stagione, specificare un numero intero o una lista di numeri interi")

      # Se season è un intero deve essere uguale a  1,2,3,4,5,6, se non lo è c'è un errore
      if  (isinstance(season, int)) & (season not in [1,2,3,4,5,6]):
         raise ValueError("Inserire una stagione valida, le stagioni vanno da 1 a 6")

      # Se è una lista controllare che sia una lista di interi e controllare che i numeri siano 1,2,3,4,5,6
      if  isinstance(season, list):
         for season_number in season:
            if not isinstance(season_number, int):
               raise TypeError("Inserire un valore valido per la stagione, specificare un numero intero")
            if season_number not in [1,2,3,4,5,6]:
               raise ValueError("Inserire una stagione valida, le stagioni vanno da 1 a 6")


      # Controllare che le guest star siano fornite in una lista
      if gueststar_apparse is not None and  not isinstance(gueststar_apparse,list):
         raise TypeError("Inserire una lista di guest star")

      # Controllare che le guest star nella lista siano stringhe
      if  isinstance(gueststar_apparse, list):
         for guest_star in gueststar_apparse:
            if not isinstance(guest_star, str):
               raise TypeError("Inserire un valore valido per la guest star, specificare una stringa")


      # Controllare che le guest star siano fornite in una lista
      if personaggi_apparsi is not None and  not isinstance(personaggi_apparsi,(list)):
         raise TypeError("Inserire una lista di personaggi")

      # Controllare che le guest star nella lista siano stringhe
      if  isinstance(personaggi_apparsi, list):
         for personaggio in personaggi_apparsi:
            if not isinstance(personaggio, str):
               raise TypeError("Inserire un valore valido per il personaggio, specificare una stringa")

      # endregion


      df = self.df
      # Lista delle colonne da estrarre dal df da mostrare all'utente come output della ricerca
      colonne_per_output = self.colonne_per_output


      # region Cambiamenti a seconda del campo di ricerca selezionato
      if campi_ricerca == "ALL":
         # Prendere il tf_idf_dataframe
         #tf_idf_dataframe=pd.read_csv(f"{self.path}/tfidf_index_ALL.csv", index_col='Unnamed: 0')
   
         tf_idf_dataframe = self.tf_idf_dataframe_all 
         word_dict = self.word_dict_all
         inverted_idx = self.inverted_idx_all   

         
      elif campi_ricerca == "title_trama":
         # Stesse operazioni dell'if, con alcuni accorgimenti specifici per questo caso
         #tf_idf_dataframe=pd.read_csv(f"{self.path}/tfidf_index_title_trama.csv", index_col='Unnamed: 0')
       
         tf_idf_dataframe = self.tf_idf_dataframe_title_trama 
         word_dict = self.word_dict_title_trama
         inverted_idx = self.inverted_idx_title_trama


        
      # endregion


      # Se non è stata inserita alcuna parola nella query restituire tutti gli episodi con i personaggi, stagione e guest star selezionate
      # Per capire se non è stata inserita alcuna parola rimuovere tutti gli spazi, se la stringa è uguale a ""
      # significa che non c'è alcuna parola
      query_str_no_spaces = re.sub( r'\s{1,}',"",query_str) # In questo modo "   " e " " e "" risultano uguali
      if query_str_no_spaces == "":
         output = self.select_by_parameters(season,personaggi_apparsi,gueststar_apparse,output_length)

         # Se l'output ha 0 righe significa che non ci sono match
         if output.shape[0] == 0:
            #raise Exception("Non ci sono puntate che corrispondono ai criteri di ricerca")
            print("Nessuna puntata ritrovata che abbia tutte le caratteristiche specificate")
            return None

         return output[colonne_per_output]




      # region Calcolo del tf-idf, cosine similarity ed estrazione puntate più simili
     

      lemmatized_query_original,useless_output = lemmatizzazione(query_str)

      # Siccome la lemmatizzazione considera differente la stessa parola con iniziale maiuscola o minuscola
      # Per ampliare la ricerca utilizzare .title() che mette tutte le parole con iniziale maiuscola
      seconda_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.title())    
   
      # fare la stessa cosa mettendole tutte minuscole
      terza_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.lower())  

      # fare la stessa cosa mettendole tutte maiuscole
      quarta_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.upper())  

      # fare la stessa cosa mettendole tutte maiuscole
      quinta_opzione_lemmatizzazione, output_inutile = lemmatizzazione(query_str.capitalize())  

      opzioni_lemmatizzazione = [   lemmatized_query_original,
                                    seconda_opzione_lemmatizzazione,
                                    terza_opzione_lemmatizzazione,
                                    quarta_opzione_lemmatizzazione,
                                    quinta_opzione_lemmatizzazione
                                 ]
      # Iterare sulle opzioni di lemmatizzazione 
      for lemmatized_query_original in opzioni_lemmatizzazione:

         # Evitare errore duplicated labels
         lemmatized_query = list(set(lemmatized_query_original))

         # Evitare key error, prende solo le parole che sono nel dictionary delle parole word, ovvero le stesse che fanno da  colonne nel dataframe dei tf-idf
         lemmatized_query = [lemmatized_word for lemmatized_word in lemmatized_query if lemmatized_word in word_dict.keys()]

         # Calcolare i tf-idf
         tf_idf_query_dataframe = self.__evaluateTFIDF__(word_dict,inverted_idx,lemmatized_query,lemmatized_query_original)
   
         # Aggiungere alla query tutte le colonne mancanti in modo da poter calcolare le cosine similarities
         tf_idf_query_dataframe=tf_idf_query_dataframe.reindex(sorted(tf_idf_dataframe.columns), axis=1) 

         # Ricodificare tutti gli NA con degli zeri
         tf_idf_query_dataframe.fillna(0,inplace=True)

         # Se tf_idf_query_dataframe.shape[0] > 0 vuol dire che c'è un match, quindi la lemmatizzazione i-esima produce risultati, perciò interrompere il for loop così da andare a calcolare le cosine similarities
         if (tf_idf_query_dataframe.shape[0] > 0):
            break 


      # Se finito il for loop il risultato ottenuto non ha trovato match interrompere e restituire None
      if (tf_idf_query_dataframe.shape[0] < 1):
         print("Le keywords indicate non hanno prodotto risultati")
         return None




      list_cosine_similarity=[]
      for i,row in tf_idf_dataframe.iterrows():      
         num=np.dot(tf_idf_query_dataframe.values.flatten(),row.values)
         den=np.linalg.norm(tf_idf_query_dataframe.values.flatten()) * np.linalg.norm(row.values)

         # Questo if statement evita le divisioni per zero.
         if den != 0:
            cosine = num / den
         else:
            cosine = 0
         #print(num,den,cosine)
         list_cosine_similarity.append(cosine)

      df["Similarity"]= list_cosine_similarity
      df= df.sort_values(by=["Similarity"],ascending=False).copy()
      df = df[df.Similarity > 0]

      
      if season != None:
         # Siccome season può essere una lista o un numero intero.
         # Ma le righe successive utilizzano sintassi per le liste, mettere il numero intero in una lista
         if isinstance(season, int):
            season = [season]

         # Restringere il dataframe df alle stagioni selezionate
         df = df[df.season.isin(season)]

      if personaggi_apparsi != None:
         df = df[(df[personaggi_apparsi] != 0).all(axis=1)]
   

      if gueststar_apparse != None:
         df = df[(df[gueststar_apparse] != 0).all(axis=1)]


      # Se l'output ha 0 righe significa che non ci sono match
      if df.shape[0] == 0:
         #raise Exception("Non ci sono puntate che corrispondono ai criteri di ricerca")
         print("Non ci sono puntate che corrispondono ai criteri di ricerca, provare altre keyword o disabilitare (se attivate) le flag su stagioni, personaggi e gueststar")
         return None


      # endregion


      df = df[colonne_per_output]
      # Mostra tutte le righe
      pd.set_option('display.max_rows', output_length)
      return df.head(output_length)