Il progetto camera cafe, mette insieme alcune delle  conoscenze acquisite durante questi anni di studio:

	- Il web scraping per l'estrazione di dati utili da wikipedia ed il loro preprocessing;			
	- Il modello whisper di openAI per la trascrizione di file audio;			
	- La teoria di una ranking engine basata su td-idf o ad una search engine più 	
	standard  basate sull' occorrenza o non occorrenza di una parola;		
	- La sintassi e le classi python che sono alla base di questo progetto.		

Ad oggi esiste una versione flask di questa search/ranking engine pronta ad essere pubblicata, tuttavia per i costi di deploy ancora non è stata pubblicata.

Per motivi di  spazio  le trascrizioni degli episodi  non sono presenti su questa repository, sono comunque presenti tutti gli script oltre ad  i vari dizionari contenenti i valori di tf-idf per ogni parola necessari al funzionamento dello script.
In questo modo basterà clonare la repository e si avrà una search engine pronta all'uso tramite il file **SearchEngine.ipynb**.


Per i più curiosi il progetto è stato sviluppato nel seguente modo.
Per prima cosa tramite un semplice script python sono stati estratti dati da wikipedia riguardanti le puntate della sitcom camera cafe (edizione Italiana): in particolare il titolo, il numero di episodio, la stagione, la trama e le guest star che appaiono.

Successivamente dai video delle puntate di camera cafe in mio possesso sono stati estratti i testi di ogni puntata.
Per fare ciò in primis sono state estratte le tracce in audio dai video tramite **pydub**.	
In seguito tramite il modello **whisper** di openAI gli audio sono stati convertiti in testo.

Una volta terminato il processo di estrazione dati tramite lo script **01_PuliziaDati.ipynb** questi dati sono stati puliti ed uniti in un unico dataframe, oltre essere effettuata una fase di feature engineering.	
Lo step finale è stato quello del calcolo dei vari indici  per far funzionare la search engine come tf-idf tramite lo script **02_Create_index.ipynb*.

La search engine è definita come una **classe** python ed è contenuta nel file **SearchEngine_class.py**.	
Questa engine possiede due modalità di utilizzo:	
	- Una ranking engine che sfrutta td-idf e cosine similarity;		
	-  Ed una più base incentrata solo sull'occorrenza/ non occorrenza delle parole.	


Entrambe le engine concedono la possibilità di restringere il campo di ricerca tramite delle flag, ad esempio la presenza di un determinato attore, oppure restringere la ricerca ad una stagione specifica.
Oltre alla funzione di limitare il numero di risultati in output o di ricercare nel titolo e  nella trama della puntata oppure su tutto il testo della puntata.



I file di trascrizione e di pulizia dati non funzioneranno in quanto i dati sono assenti, mentre la search engine funziona ed è scaricabile,  per poterla far funzionare potrebbe essere necessario cambiare alcuni percorsi file (i percorsi file sono sempre modificabili all'inizio dello script) e ovviamente è necessario avere python ed i pacchetti necessari installati sul proprio pc.

Pacchetti necessari:		
	- Nel file **whisper.yml** si possono trovare i pacchetti necessari per utilizzare il modello whisper di open AI	
	

Mentre per l'utilizzo della search engine sono necessari il pacchetto:		
	 - **nltk**		
	 - **spacy** in particolare di spacy `'it_core_news_sm' 		
	 	(python -m spacy download it_core_news_sm)		
	 - **sklearn**		
	 - **pyarrow** pacchetti per maneggiare 		
	




