import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_outliers_iterative_manual(input_file, k=5, n_init=50, sigma_final=3.0):
    # 1. Φόρτωση
    try:
        df = pd.read_csv(input_file, header=None, names=['x', 'y'])
    except FileNotFoundError:
        print(f"Σφάλμα: Δεν βρέθηκε το αρχείο {input_file}")
        return

    # 2. MANUAL SCALING (Η σωστή μέθοδος για τη γεωμετρία σου)
    # Βρίσκουμε πόσες φορές μεγαλύτερο είναι το Y από το X
    max_x = df['x'].max()
    max_y = df['y'].max()
    scaling_factor = max_y / max_x
    
    # Φτιάχνουμε το scaled dataset διαιρώντας το Y
    df_scaled = df.copy()
    df_scaled['y'] = df['y'] / scaling_factor
    
    # Πλέον το df_scaled έχει x και y στην ίδια κλίμακα (περίπου 0-5)

    # --- ΒΗΜΑ 1: Αρχικό K-Means (Βρώμικο) ---
    kmeans_dirty = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    labels_dirty = kmeans_dirty.fit_predict(df_scaled[['x', 'y']])
    centroids_dirty = kmeans_dirty.cluster_centers_

    # Υπολογισμός αποστάσεων από τα "βρώμικα" κέντρα
    dists_dirty = []
    for i, row in df_scaled.iterrows():
        c = centroids_dirty[labels_dirty[i]]
        # Ευκλείδεια απόσταση στο scaled χώρο
        dist = np.linalg.norm(np.array([row['x'], row['y']]) - c)
        dists_dirty.append(dist)
    
    # --- ΒΗΜΑ 2: Φιλτράρισμα (Refinement) ---
    # Κρατάμε ΜΟΝΟ το 30% των πιο κοντινών σημείων (τον πυρήνα)
    threshold_temp = np.percentile(dists_dirty, 30) 
    clean_mask = np.array(dists_dirty) <= threshold_temp
    
    df_clean_temp = df_scaled[clean_mask]

    # --- ΒΗΜΑ 3: K-Means στα ΚΑΘΑΡΑ δεδομένα ---
    kmeans_clean = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    kmeans_clean.fit(df_clean_temp[['x', 'y']])
    
    # Αυτά είναι τα "Τέλεια Κέντρα" (στο scaled χώρο)
    perfect_centroids = kmeans_clean.cluster_centers_

    # --- ΒΗΜΑ 4: Τελική Ταξινόμηση ---
    # Χρησιμοποιούμε τα τέλεια κέντρα για να κρίνουμε ΟΛΑ τα σημεία
    final_labels = kmeans_clean.predict(df_scaled[['x', 'y']])
    
    final_distances = []
    for i, row in df_scaled.iterrows():
        c = perfect_centroids[final_labels[i]]
        dist = np.linalg.norm(np.array([row['x'], row['y']]) - c)
        final_distances.append(dist)
    
    # Τελικό Thresholding (Z-Score)
    mean_d = np.mean(final_distances)
    std_d = np.std(final_distances)
    final_threshold = mean_d + (sigma_final * std_d)
    
    outliers_mask = np.array(final_distances) > final_threshold
    
    # --- Plotting ---
    df['label'] = 'Normal'
    df.loc[outliers_mask, 'label'] = 'Outlier'
    df['cluster'] = final_labels

    plt.figure(figsize=(12, 7))
    
    # Plot Normal (Χρησιμοποιούμε τις ΑΡΧΙΚΕΣ τιμές x, y για να φαίνεται σωστά το γράφημα)
    normal = df[df['label'] == 'Normal']
    plt.scatter(normal['x'], normal['y'], c=normal['cluster'], cmap='tab10', s=20, alpha=0.6, label='Normal')
    
    # Plot Outliers
    outliers = df[df['label'] == 'Outlier']
    plt.scatter(outliers['x'], outliers['y'], c='red', marker='x', s=80, label='Outliers')
    
    # Plot Centroids
    # ΠΡΟΣΟΧΗ: Τα centroids είναι scaled. Πρέπει να πολλαπλασιάσουμε το y με το factor για να τα δούμε
    centroids_original_x = perfect_centroids[:, 0]
    centroids_original_y = perfect_centroids[:, 1] * scaling_factor
    
    plt.scatter(centroids_original_x, centroids_original_y, 
                c='black', marker='*', s=300, label='Refined Centroids', edgecolors='white')

    plt.title(f'Final Method: Manual Scaling + Iterative K-Means\nFile: {input_file} | Sigma: {sigma_final}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"--- {input_file} ---")
    print(f"Outliers detected: {sum(outliers_mask)}")
    print(f"Scaling Factor used: Y divided by {scaling_factor:.2f}")

# ΕΚΤΕΛΕΣΗ
find_outliers_iterative_manual('data/clean_data_a.csv', k=5, sigma_final=3.0)
find_outliers_iterative_manual('data/clean_data_b.csv', k=5, sigma_final=3.0)